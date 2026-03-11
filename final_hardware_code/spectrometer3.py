import time
import sys
import math
from as7343 import AS7343

# ==========================================
# CONFIGURATION
# ==========================================
GAIN = 512
INTEGRATION_TIME = 200
CALIBRATION_SAMPLES = 12   # samples per material during calibration
SCAN_SAMPLES = 6           # readings averaged per scan
LOW_CONFIDENCE_THRESHOLD = 15  # % — below this, warn the user

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
    Read all available spectral channels from both banks,
    then normalize by total intensity.
    Returns a dict of {channel_name: normalized_fraction}.
    """
    combined = {}

    # Only bank 1 — bank 0 hangs on most AS7343 setups
    sensor.bank_select(1)
    time.sleep(0.25)
    data = sensor.get_data()[0]
    for ch, val in data.items():
        if ch not in EXCLUDE_CHANNELS:
            combined[ch] = max(float(val), 0.0)

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


# ==========================================
# CALIBRATION: BOTH MATERIALS
# ==========================================
def calibrate_material(label, samples=CALIBRATION_SAMPLES):
    """
    Capture a stable spectral fingerprint for one material.
    Rejects outliers (readings more than 2 std devs from mean distance).
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

    # Outlier rejection: drop readings far from the group's centroid
    avg = average_fingerprints(fingerprints)
    distances = [euclidean_distance(fp, avg) for fp in fingerprints]
    mean_d = sum(distances) / len(distances)
    std_d = math.sqrt(sum((d - mean_d) ** 2 for d in distances) / len(distances))
    threshold = mean_d + 2 * std_d

    clean = [fp for fp, d in zip(fingerprints, distances) if d <= threshold]
    dropped = len(fingerprints) - len(clean)

    if dropped:
        print(f"  (Dropped {dropped} noisy samples)")

    result = average_fingerprints(clean)
    print(f"  ✓ {label} calibrated from {len(clean)} clean samples.")
    return result


def run_calibration():
    print("\n" + "=" * 42)
    print("  CALIBRATION MODE")
    print("  You'll calibrate BOTH materials so the")
    print("  system classifies by similarity — not")
    print("  by arbitrary thresholds.")
    print("=" * 42)

    plastic_fp = None
    glass_fp = None

    while plastic_fp is None:
        plastic_fp = calibrate_material("PLASTIC (or air)")

    while glass_fp is None:
        glass_fp = calibrate_material("GLASS")

    # Sanity check: are the two fingerprints actually different?
    sep = euclidean_distance(plastic_fp, glass_fp)
    print(f"\n  Calibration separation: {sep:.4f}")
    if sep < 0.01:
        print("  ⚠ WARNING: Materials look very similar to the sensor.")
        print("    Results may be unreliable. Try improving lighting")
        print("    or repositioning the sensor.")
    else:
        print("  ✓ Good separation — sensor can distinguish these materials.")

    return plastic_fp, glass_fp


# ==========================================
# SCANNING & CLASSIFICATION
# ==========================================
def scan_and_classify(plastic_fp, glass_fp, samples=SCAN_SAMPLES):
    """
    Take several readings, average them, then classify by nearest fingerprint.
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

    scan_fp = average_fingerprints(fingerprints)

    d_plastic = euclidean_distance(scan_fp, plastic_fp)
    d_glass = euclidean_distance(scan_fp, glass_fp)

    # Confidence: how decisively it matches one vs the other
    # 0% = equidistant (totally ambiguous), 100% = perfect match to one only
    total = d_plastic + d_glass
    confidence = abs(d_plastic - d_glass) / total * 100 if total > 0 else 0

    result = "PLASTIC" if d_plastic <= d_glass else "GLASS"

    debug = {
        "d_plastic": d_plastic,
        "d_glass": d_glass,
        "samples_used": len(fingerprints),
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

    plastic_fp, glass_fp = run_calibration()

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
            plastic_fp, glass_fp = run_calibration()
            print("\nSystem ready.")
            continue

        print("  Scanning...", end="", flush=True)
        result, confidence, debug = scan_and_classify(plastic_fp, glass_fp)
        print(" done.\n")

        # Confidence warning
        if result == "ERROR":
            print("  ✗ Sensor read failed. Check hardware.")
            continue

        conf_symbol = "✓✓" if confidence > 40 else ("✓" if confidence > LOW_CONFIDENCE_THRESHOLD else "⚠")

        print(f"  {'=' * 38}")
        print(f"  RESULT:      {result}")
        print(f"  Confidence:  {confidence:.1f}%  {conf_symbol}")
        print(f"  ── debug ──────────────────────────")
        print(f"  Dist→PLASTIC: {debug['d_plastic']:.5f}")
        print(f"  Dist→GLASS:   {debug['d_glass']:.5f}")
        print(f"  Samples used: {debug['samples_used']}")

        if confidence < LOW_CONFIDENCE_THRESHOLD:
            print(f"\n  ⚠ LOW CONFIDENCE — re-scan or recalibrate (c)")

        print(f"  {'=' * 38}")


if __name__ == "__main__":
    main()