import time
import sys
import math
import json
import os
from as7343 import AS7343

# ==========================================
# CONFIGURATION
# ==========================================
GAIN = 2048
INTEGRATION_TIME = 200
CALIBRATION_SAMPLES = 12       # samples per material during calibration
SCAN_SAMPLES = 6               # readings averaged per scan
LOW_CONFIDENCE_THRESHOLD = 15  # % — below this, warn the user
CALIB_FILE = "calibration.json"

# Channels to exclude from fingerprint (non-spectral)
EXCLUDE_CHANNELS = {'flicker', 'clear'}

try:
    sensor = AS7343()
    sensor.set_gain(GAIN)
    sensor.set_integration_time(INTEGRATION_TIME)
except Exception as e:
    print(f"Hardware Error: {e}")
    sys.exit()


# ==========================================
# CORE: FULL NORMALIZED SPECTRAL FINGERPRINT
# ==========================================
# KEY INSIGHT: By dividing every channel by the SUM of all channels,
# absolute intensity cancels out. What remains is the spectral *shape*,
# which is a material property — not an environmental one.
# Glass absorbs UV differently from plastic across MULTIPLE bands,
# so the full shape is a far more stable signature than any single ratio.

def get_spectral_fingerprint():
    """
    Read all available spectral channels from BOTH banks (0 and 1).
    Bank 0 contains UV/violet channels — the best discriminators between
    glass (UV-absorbing) and plastic (UV-transmitting).
    Normalizes by total intensity so only spectral shape matters.
    Returns a dict of {channel_name: normalized_fraction}, or None on failure.
    """
    combined = {}

    for bank in [0, 1]:
        sensor.bank_select(bank)
        time.sleep(0.30)  # allow bank switch to settle
        try:
            data = sensor.get_data()[0]
            for ch, val in data.items():
                if ch not in EXCLUDE_CHANNELS:
                    combined[ch] = max(float(val), 0.0)
        except Exception:
            pass  # gracefully skip a hanging or unavailable bank

    if not combined:
        return None

    total = sum(combined.values())
    if total < 1:
        return None  # sensor returned nothing useful

    # Normalize: each value is now a fraction of total light
    # This is immune to brightness changes — only spectral shape matters
    return {k: v / total for k, v in combined.items()}


def average_fingerprints(fingerprints):
    """Average a list of fingerprint dicts into one."""
    keys = fingerprints[0].keys()
    return {k: sum(fp[k] for fp in fingerprints) / len(fingerprints) for k in keys}


def euclidean_distance(fp_a, fp_b):
    """
    Euclidean distance between two normalized fingerprints.
    Lower = more similar spectral shape.
    """
    common = set(fp_a.keys()) & set(fp_b.keys())
    return math.sqrt(sum((fp_a[k] - fp_b[k]) ** 2 for k in common))


def weighted_distance(fp_scan, material_profile):
    """
    Mahalanobis-style weighted distance.
    Channels with low variance during calibration are more reliable
    and contribute more to the distance score.
    This makes classification far more robust to noisy channels.
    """
    mean = material_profile["mean"]
    std  = material_profile["std"]
    common = set(fp_scan.keys()) & set(mean.keys())
    return math.sqrt(sum(
        ((fp_scan[k] - mean[k]) / (std[k] + 1e-6)) ** 2
        for k in common
    ))


def reject_outliers(fingerprints):
    """
    Remove readings more than 2 std deviations from the group centroid.
    Returns (clean_list, n_dropped).
    """
    if len(fingerprints) < 3:
        return fingerprints, 0

    avg = average_fingerprints(fingerprints)
    distances = [euclidean_distance(fp, avg) for fp in fingerprints]
    mean_d = sum(distances) / len(distances)
    std_d = math.sqrt(sum((d - mean_d) ** 2 for d in distances) / len(distances))
    threshold = mean_d + 2 * std_d

    clean = [fp for fp, d in zip(fingerprints, distances) if d <= threshold]
    dropped = len(fingerprints) - len(clean)
    return clean, dropped


# ==========================================
# CALIBRATION PERSISTENCE
# ==========================================

def save_calibration(plastic_profile, glass_profile):
    """Save calibration profiles to disk so recalibration isn't needed every restart."""
    with open(CALIB_FILE, "w") as f:
        json.dump({"plastic": plastic_profile, "glass": glass_profile}, f, indent=2)
    print("  ✓ Calibration saved to disk.")


def load_calibration():
    """Load saved calibration profiles from disk. Returns (None, None) if not found."""
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
    Capture a stable spectral fingerprint profile for one material.
    Returns a profile dict containing:
      - mean:  average normalized fingerprint
      - std:   per-channel standard deviation (used for weighted distance)
      - n:     number of clean samples used
    Rejects outliers (readings more than 2 std devs from the group centroid).
    """
    print(f"\n  → Place {label} directly on the sensor.")
    input("    Press Enter when ready...")
    print(f"  Collecting {samples} samples for {label}...")

    fingerprints = []
    for i in range(samples):
        fp = get_spectral_fingerprint()
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
        print("  ERROR: Too many samples were rejected. Check sensor placement.")
        return None

    mean = average_fingerprints(clean)

    # Per-channel standard deviation — captures how stable each channel is
    # Channels with low std are reliable discriminators; high std = noisy
    keys = mean.keys()
    std = {
        k: math.sqrt(sum((fp[k] - mean[k]) ** 2 for fp in clean) / len(clean))
        for k in keys
    }

    print(f"  ✓ {label} calibrated from {len(clean)} clean samples.")
    return {"mean": mean, "std": std, "n": len(clean)}


def run_calibration():
    print("\n" + "=" * 42)
    print("  CALIBRATION MODE")
    print("  You'll calibrate BOTH materials so the")
    print("  system classifies by similarity — not")
    print("  by arbitrary thresholds.")
    print("=" * 42)

    plastic_profile = None
    glass_profile = None

    while plastic_profile is None:
        plastic_profile = calibrate_material("PLASTIC (or air)")

    while glass_profile is None:
        glass_profile = calibrate_material("GLASS")

    # Sanity check: are the two fingerprints actually different?
    sep = euclidean_distance(plastic_profile["mean"], glass_profile["mean"])
    print(f"\n  Calibration separation: {sep:.4f}")
    if sep < 0.01:
        print("  ⚠ WARNING: Materials look very similar to the sensor.")
        print("    Results may be unreliable. Try improving lighting")
        print("    or repositioning the sensor.")
    else:
        print("  ✓ Good separation — sensor can distinguish these materials.")

    save_calibration(plastic_profile, glass_profile)
    return plastic_profile, glass_profile


# ==========================================
# SCANNING & CLASSIFICATION
# ==========================================

def compute_confidence(d_plastic, d_glass, plastic_profile, glass_profile):
    """
    Compute a meaningful confidence score that accounts for:
      1. How decisive the margin is (d_plastic vs d_glass difference)
      2. How close the winner is in absolute terms (not just relative)

    Both are normalized against the calibration spread so the scale is meaningful.
    Returns a float 0–100.
    """
    calibration_spread = euclidean_distance(
        plastic_profile["mean"], glass_profile["mean"]
    )
    if calibration_spread < 1e-6:
        return 0.0

    margin = abs(d_plastic - d_glass)
    winner_dist = min(d_plastic, d_glass)

    # How close is the match relative to the calibration spread?
    proximity_score = max(0.0, 1.0 - (winner_dist / calibration_spread))

    # How decisive is the margin relative to the calibration spread?
    margin_score = min(1.0, margin / calibration_spread)

    # Weighted blend: proximity matters more than margin
    return (proximity_score * 0.6 + margin_score * 0.4) * 100


def scan_and_classify(plastic_profile, glass_profile, samples=SCAN_SAMPLES):
    """
    Take several readings, reject outliers, average them, then classify
    by nearest calibrated profile using weighted (Mahalanobis-style) distance.

    Also detects "unknown/nothing" when the sample is far from both materials.

    Returns (result_label, confidence_percent, debug_info).
    """
    fingerprints = []
    for _ in range(samples):
        fp = get_spectral_fingerprint()
        if fp is not None:
            fingerprints.append(fp)
        time.sleep(0.08)

    if not fingerprints:
        return "ERROR", 0, {}

    # Reject scan outliers — same logic as calibration
    clean, dropped = reject_outliers(fingerprints)
    if not clean:
        return "ERROR", 0, {}

    scan_fp = average_fingerprints(clean)

    d_plastic = weighted_distance(scan_fp, plastic_profile)
    d_glass   = weighted_distance(scan_fp, glass_profile)

    # Detect "nothing on sensor" — if scan is far from BOTH calibrated
    # materials (relative to how different those materials are from each other),
    # it's likely air, an unknown material, or a misread.
    calib_spread = euclidean_distance(
        plastic_profile["mean"], glass_profile["mean"]
    )
    if min(d_plastic, d_glass) > calib_spread * 1.5:
        debug = {
            "d_plastic": d_plastic,
            "d_glass": d_glass,
            "samples_used": len(clean),
            "samples_dropped": dropped,
        }
        return "UNKNOWN", 0, debug

    confidence = compute_confidence(d_plastic, d_glass, plastic_profile, glass_profile)
    result = "PLASTIC" if d_plastic <= d_glass else "GLASS"

    debug = {
        "d_plastic": d_plastic,
        "d_glass": d_glass,
        "samples_used": len(clean),
        "samples_dropped": dropped,
    }
    return result, confidence, debug


# ==========================================
# MAIN LOOP
# ==========================================

def main():
    print("\n" + "=" * 42)
    print("  GLASS vs PLASTIC CLASSIFIER")
    print("  Full Spectral Fingerprinting Mode")
    print("=" * 42)
    print("\nCommands:")
    print("  Enter  → scan")
    print("  c      → recalibrate")
    print("  q      → quit")

    # Try to load saved calibration first — recalibrate only if needed
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

        print("  Scanning...", end="", flush=True)
        result, confidence, debug = scan_and_classify(plastic_profile, glass_profile)
        print(" done.\n")

        if result == "ERROR":
            print("  ✗ Sensor read failed. Check hardware.")
            continue

        if result == "UNKNOWN":
            print(f"  {'=' * 38}")
            print(f"  RESULT:      ⚠ UNKNOWN / NOTHING DETECTED")
            print(f"  The reading is too far from both calibrated")
            print(f"  materials. Check sensor placement.")
            print(f"  ── debug ──────────────────────────")
            print(f"  Dist→PLASTIC: {debug['d_plastic']:.5f}")
            print(f"  Dist→GLASS:   {debug['d_glass']:.5f}")
            print(f"  Samples used: {debug['samples_used']}  "
                  f"(dropped: {debug['samples_dropped']})")
            print(f"  {'=' * 38}")
            continue

        conf_symbol = "✓✓" if confidence > 40 else ("✓" if confidence > LOW_CONFIDENCE_THRESHOLD else "⚠")

        print(f"  {'=' * 38}")
        print(f"  RESULT:      {result}")
        print(f"  Confidence:  {confidence:.1f}%  {conf_symbol}")
        print(f"  ── debug ──────────────────────────")
        print(f"  Dist→PLASTIC: {debug['d_plastic']:.5f}")
        print(f"  Dist→GLASS:   {debug['d_glass']:.5f}")
        print(f"  Samples used: {debug['samples_used']}  "
              f"(dropped: {debug['samples_dropped']})")

        if confidence < LOW_CONFIDENCE_THRESHOLD:
            print(f"\n  ⚠ LOW CONFIDENCE — re-scan or recalibrate (c)")

        print(f"  {'=' * 38}")


if __name__ == "__main__":
    main()