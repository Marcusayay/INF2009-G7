import time
import sys
import math
import json
import os
from as7343 import AS7343

# ==========================================
# CONFIGURATION
# ==========================================
INTEGRATION_TIME         = 200
CALIBRATION_SAMPLES      = 12    # samples per material during calibration
SCAN_SAMPLES             = 6     # readings averaged per scan
LOW_CONFIDENCE_THRESHOLD = 15    # % — below this, warn the user
CALIB_FILE               = "calibration.json"

# Channels to exclude from fingerprint (non-spectral)
EXCLUDE_CHANNELS = {'flicker', 'clear'}

# ── Auto-gain settings ──────────────────────────────────────────────────────
# All gain steps supported by the AS7343, ordered lowest → highest.
# Auto-gain walks UP until the sensor is about to saturate, then steps
# back one level — maximum SNR without clipping.
VALID_GAINS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

# 16-bit ADC: full-scale = 65535. We treat anything above 90 % as saturated
# so we stay safely below clipping even with slight flicker.
ADC_MAX              = 65535
SATURATION_PCT       = 0.90
SATURATION_THRESHOLD = int(ADC_MAX * SATURATION_PCT)   # ≈ 58 981

# Minimum total counts to consider a reading valid.
# Below this the SNR is too poor to trust the spectral shape.
MIN_TOTAL_COUNTS = 500
# ────────────────────────────────────────────────────────────────────────────

try:
    sensor = AS7343()
    sensor.set_integration_time(INTEGRATION_TIME)

except Exception as e:
    print(f"Hardware Error: {e}")
    sys.exit()


# ==========================================
# AUTO-GAIN
# ==========================================

def _read_raw_all_banks():
    """
    Read raw (un-normalised) counts from both banks.
    Returns a flat dict {channel: raw_count}, or {} on total failure.
    Bank 0 holds the UV/violet channels — the strongest glass/plastic
    discriminators — so both banks must be read.
    """
    combined = {}
    for bank in [0, 1]:
        sensor.bank_select(bank)
        time.sleep(0.25)
        try:
            data = sensor.get_data()[0]
            for ch, val in data.items():
                if ch not in EXCLUDE_CHANNELS:
                    combined[ch] = max(float(val), 0.0)
        except Exception:
            pass  # skip a hanging or unavailable bank
    return combined


def auto_set_gain(verbose=True):
    """
    Walk VALID_GAINS from lowest to highest.
    Stop one step before any channel saturates, or stay at the highest
    gain if saturation is never reached (very dim conditions).

    Algorithm
    ---------
    For each candidate gain:
      1. Apply it and wait briefly for the sensor to settle.
      2. Take one raw reading from both banks.
      3. If any channel hits SATURATION_THRESHOLD → the *previous* gain
         was safe; restore it and return.
      4. Otherwise record this as the new best-safe gain and go higher.
    If we exhaust all gains without clipping, the max gain is used and a
    warning is shown if total counts are still below MIN_TOTAL_COUNTS.

    Returns the gain value that was ultimately applied.
    """
    if verbose:
        print("  [Auto-gain] Probing...", end="", flush=True)

    best_gain = VALID_GAINS[0]

    for gain in VALID_GAINS:
        sensor.set_gain(gain)
        time.sleep(0.15)   # let the sensor settle after each gain change

        raw = _read_raw_all_banks()
        if not raw:
            continue

        peak  = max(raw.values())
        total = sum(raw.values())

        if peak >= SATURATION_THRESHOLD:
            # This gain clips — step back to the last safe level
            sensor.set_gain(best_gain)
            if verbose:
                print(f" set gain={best_gain}  "
                      f"(peak {peak:.0f} clipped at gain={gain})")
            return best_gain

        best_gain = gain  # safe so far — try the next step up

    # Reached the top without saturating
    sensor.set_gain(best_gain)
    total_at_max = sum(_read_raw_all_banks().values())

    if verbose:
        if total_at_max < MIN_TOTAL_COUNTS:
            print(f" set gain={best_gain}  "
                  f"⚠ weak signal ({total_at_max:.0f} counts) — check placement")
        else:
            print(f" set gain={best_gain}  (no saturation reached)")

    return best_gain


# ==========================================
# CORE: FULL NORMALIZED SPECTRAL FINGERPRINT
# ==========================================

def get_spectral_fingerprint(run_agc=False):
    """
    Read all spectral channels from both banks and normalize by total
    intensity — only spectral *shape* survives, not brightness.

    run_agc=True  → re-run auto_set_gain before reading (used once per
                    calibration or scan burst to lock in the right gain).
    run_agc=False → trust the currently-set gain (used for rapid successive
                    reads within the same burst so gain is consistent).

    Returns {channel: normalized_fraction} or None on failure.
    """
    if run_agc:
        auto_set_gain(verbose=False)

    combined = _read_raw_all_banks()
    if not combined:
        return None

    total = sum(combined.values())
    if total < MIN_TOTAL_COUNTS:
        return None

    return {k: v / total for k, v in combined.items()}


def average_fingerprints(fingerprints):
    """Average a list of fingerprint dicts into one."""
    keys = fingerprints[0].keys()
    return {k: sum(fp[k] for fp in fingerprints) / len(fingerprints) for k in keys}


def euclidean_distance(fp_a, fp_b):
    """Standard Euclidean distance between two normalised fingerprints."""
    common = set(fp_a.keys()) & set(fp_b.keys())
    return math.sqrt(sum((fp_a[k] - fp_b[k]) ** 2 for k in common))


def weighted_distance(fp_scan, material_profile):
    """
    Mahalanobis-style weighted distance.
    Channels that were stable during calibration (low std) contribute
    more to the score; noisy channels are down-weighted automatically.
    """
    mean   = material_profile["mean"]
    std    = material_profile["std"]
    common = set(fp_scan.keys()) & set(mean.keys())
    return math.sqrt(sum(
        ((fp_scan[k] - mean[k]) / (std[k] + 1e-6)) ** 2
        for k in common
    ))


def reject_outliers(fingerprints):
    """
    Drop readings more than 2 std devs from the group centroid.
    Returns (clean_list, n_dropped).
    """
    if len(fingerprints) < 3:
        return fingerprints, 0

    avg       = average_fingerprints(fingerprints)
    distances = [euclidean_distance(fp, avg) for fp in fingerprints]
    mean_d    = sum(distances) / len(distances)
    std_d     = math.sqrt(sum((d - mean_d) ** 2 for d in distances) / len(distances))
    threshold = mean_d + 2 * std_d

    clean   = [fp for fp, d in zip(fingerprints, distances) if d <= threshold]
    dropped = len(fingerprints) - len(clean)
    return clean, dropped


# ==========================================
# CALIBRATION PERSISTENCE
# ==========================================

def save_calibration(plastic_profile, glass_profile):
    """Persist calibration profiles to disk — no recalibration on restart."""
    with open(CALIB_FILE, "w") as f:
        json.dump({"plastic": plastic_profile, "glass": glass_profile}, f, indent=2)
    print("  ✓ Calibration saved to disk.")


def load_calibration():
    """Load saved profiles from disk. Returns (None, None) if unavailable."""
    if not os.path.exists(CALIB_FILE):
        return None, None
    try:
        with open(CALIB_FILE) as f:
            data = json.load(f)
        print("  ✓ Loaded saved calibration from disk.")
        return data["plastic"], data["glass"]
    except Exception as e:
        print(f"  ⚠ Could not load calibration ({e}). Recalibrating...")
        return None, None


# ==========================================
# CALIBRATION: BOTH MATERIALS
# ==========================================

def calibrate_material(label, samples=CALIBRATION_SAMPLES):
    """
    Collect a stable spectral fingerprint profile for one material.

    Auto-gain runs once with the actual material in place so the gain
    is tuned to that specific surface's reflectance — not some default.
    All subsequent reads in the burst use that fixed gain, keeping the
    normalised fingerprints internally consistent.

    Returns {mean, std, n, gain_used} or None on failure.
    """
    print(f"\n  → Place {label} directly on the sensor.")
    input("    Press Enter when ready...")

    print(f"  Setting optimal gain for {label}...", end="", flush=True)
    gain_used = auto_set_gain(verbose=False)
    print(f" gain={gain_used}")

    print(f"  Collecting {samples} samples...")

    fingerprints = []
    for _ in range(samples):
        fp = get_spectral_fingerprint(run_agc=False)  # gain already locked
        if fp is not None:
            fingerprints.append(fp)
        time.sleep(0.12)

    if len(fingerprints) < 3:
        print("  ERROR: Not enough valid readings. Try again.")
        return None

    clean, dropped = reject_outliers(fingerprints)
    if dropped:
        print(f"  (Dropped {dropped} noisy samples)")

    if len(clean) < 2:
        print("  ERROR: Too many samples rejected. Check sensor placement.")
        return None

    mean = average_fingerprints(clean)
    std  = {
        k: math.sqrt(sum((fp[k] - mean[k]) ** 2 for fp in clean) / len(clean))
        for k in mean.keys()
    }

    print(f"  ✓ {label} calibrated from {len(clean)} samples  (gain={gain_used}).")
    return {"mean": mean, "std": std, "n": len(clean), "gain_used": gain_used}


def run_calibration():
    print("\n" + "=" * 42)
    print("  CALIBRATION MODE")
    print("  Auto-gain optimises sensitivity for each")
    print("  material individually before sampling.")
    print("=" * 42)

    plastic_profile = None
    glass_profile   = None

    while plastic_profile is None:
        plastic_profile = calibrate_material("PLASTIC (or air)")

    while glass_profile is None:
        glass_profile = calibrate_material("GLASS")

    sep = euclidean_distance(plastic_profile["mean"], glass_profile["mean"])
    print(f"\n  Calibration separation: {sep:.4f}")
    if sep < 0.01:
        print("  ⚠ WARNING: Materials look very similar to the sensor.")
        print("    Results may be unreliable. Improve lighting or reposition sensor.")
    else:
        print("  ✓ Good separation — sensor can distinguish these materials.")

    save_calibration(plastic_profile, glass_profile)
    return plastic_profile, glass_profile


# ==========================================
# SCANNING & CLASSIFICATION
# ==========================================

def compute_confidence(d_plastic, d_glass, plastic_profile, glass_profile):
    """
    Confidence score normalized against the calibration spread so the
    number is meaningful across different lighting conditions.

    Combines:
      - proximity_score: how close the scan is to the winning material
      - margin_score:    how decisively it beat the other material

    Returns 0–100 %.
    """
    spread = euclidean_distance(plastic_profile["mean"], glass_profile["mean"])
    if spread < 1e-6:
        return 0.0

    margin      = abs(d_plastic - d_glass)
    winner_dist = min(d_plastic, d_glass)

    proximity_score = max(0.0, 1.0 - (winner_dist / spread))
    margin_score    = min(1.0,  margin / spread)

    return (proximity_score * 0.6 + margin_score * 0.4) * 100


def scan_and_classify(plastic_profile, glass_profile, samples=SCAN_SAMPLES):
    """
    Full scan pipeline:
      1. Auto-gain probe with whatever is on the sensor.
      2. Collect `samples` readings at that fixed gain.
      3. Reject outliers.
      4. Classify by weighted distance to each calibrated profile.
      5. Return UNKNOWN if the scan is too far from both profiles.

    Returns (result_label, confidence_percent, debug_dict).
    """
    # Step 1 — AGC: find the right gain for this specific sample
    gain_used = auto_set_gain(verbose=True)

    # Step 2 — collect readings at the locked gain
    fingerprints = []
    for _ in range(samples):
        fp = get_spectral_fingerprint(run_agc=False)
        if fp is not None:
            fingerprints.append(fp)
        time.sleep(0.08)

    if not fingerprints:
        return "ERROR", 0, {}

    # Step 3 — reject outliers
    clean, dropped = reject_outliers(fingerprints)
    if not clean:
        return "ERROR", 0, {}

    scan_fp = average_fingerprints(clean)

    # Step 4 — weighted distance to each material
    d_plastic = weighted_distance(scan_fp, plastic_profile)
    d_glass   = weighted_distance(scan_fp, glass_profile)

    # Step 5 — "nothing on sensor" guard
    calib_spread = euclidean_distance(
        plastic_profile["mean"], glass_profile["mean"]
    )
    if min(d_plastic, d_glass) > calib_spread * 1.5:
        return "UNKNOWN", 0, {
            "d_plastic":      d_plastic,
            "d_glass":        d_glass,
            "samples_used":   len(clean),
            "samples_dropped": dropped,
            "gain_used":      gain_used,
        }

    confidence = compute_confidence(d_plastic, d_glass, plastic_profile, glass_profile)
    result     = "PLASTIC" if d_plastic <= d_glass else "GLASS"

    return result, confidence, {
        "d_plastic":       d_plastic,
        "d_glass":         d_glass,
        "samples_used":    len(clean),
        "samples_dropped": dropped,
        "gain_used":       gain_used,
    }


# ==========================================
# MAIN LOOP
# ==========================================

def main():
    print("\n" + "=" * 42)
    print("  GLASS vs PLASTIC CLASSIFIER")
    print("  Spectral Fingerprinting + Auto-Gain")
    print("=" * 42)
    print("\nCommands:")
    print("  Enter  → scan")
    print("  c      → recalibrate")
    print("  g      → manual auto-gain probe")
    print("  q      → quit")

    # Load saved calibration or run fresh calibration
    plastic_profile, glass_profile = load_calibration()
    if plastic_profile is None or glass_profile is None:
        plastic_profile, glass_profile = run_calibration()

    print("\n" + "=" * 42)
    print("  SYSTEM READY — press Enter to scan")
    print("=" * 42)

    while True:
        try:
            cmd = input("\nScan? ").lower().strip()
        except KeyboardInterrupt:
            print("\nExit.")
            break

        if cmd == 'q':
            break

        elif cmd == 'c':
            plastic_profile, glass_profile = run_calibration()
            print("\nSystem ready.")
            continue

        elif cmd == 'g':
            print("  Running auto-gain probe on current surface...")
            g = auto_set_gain(verbose=True)
            print(f"  Gain set to {g}.")
            continue

        print("  Scanning...", end="", flush=True)
        result, confidence, debug = scan_and_classify(plastic_profile, glass_profile)
        print(" done.\n")

        if result == "ERROR":
            print("  ✗ Sensor read failed. Check hardware.")
            continue

        if result == "UNKNOWN":
            print(f"  {'=' * 38}")
            print(f"  RESULT:      ⚠ UNKNOWN / NOTHING DETECTED")
            print(f"  Reading is far from both calibrated materials.")
            print(f"  Check sensor placement.")
            print(f"  ── debug ──────────────────────────")
            print(f"  Gain used:    {debug.get('gain_used', '?')}")
            print(f"  Dist→PLASTIC: {debug['d_plastic']:.5f}")
            print(f"  Dist→GLASS:   {debug['d_glass']:.5f}")
            print(f"  Samples used: {debug['samples_used']}  "
                  f"(dropped: {debug['samples_dropped']})")
            print(f"  {'=' * 38}")
            continue

        conf_symbol = ("✓✓" if confidence > 40
                       else "✓" if confidence > LOW_CONFIDENCE_THRESHOLD
                       else "⚠")

        print(f"  {'=' * 38}")
        print(f"  RESULT:      {result}")
        print(f"  Confidence:  {confidence:.1f}%  {conf_symbol}")
        print(f"  ── debug ──────────────────────────")
        print(f"  Gain used:    {debug['gain_used']}")
        print(f"  Dist→PLASTIC: {debug['d_plastic']:.5f}")
        print(f"  Dist→GLASS:   {debug['d_glass']:.5f}")
        print(f"  Samples used: {debug['samples_used']}  "
              f"(dropped: {debug['samples_dropped']})")

        if confidence < LOW_CONFIDENCE_THRESHOLD:
            print(f"\n  ⚠ LOW CONFIDENCE — re-scan or recalibrate (c)")

        print(f"  {'=' * 38}")


if __name__ == "__main__":
    main()