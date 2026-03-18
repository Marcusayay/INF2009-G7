import time
import sys
from as7343 import AS7343

# ==========================================
# REFINED THRESHOLDS
# ==========================================
# We define a "Safe Zone" around the baseline.
# If the difference is within +/- 0.02, we call it PLASTIC (it's like air).
# If the difference drops below -0.02, it's GLASS.
SAFE_ZONE_MARGIN = 0.02 

try:
    sensor = AS7343()
    sensor.set_gain(64)
    sensor.set_integration_time(200)
except Exception as e:
    print(f"Hardware Error: {e}"); sys.exit()

def get_spectral_data():
    sensor.bank_select(1)
    time.sleep(0.25)
    d1 = sensor.get_data()[0]
    f1_uv = d1['fz']    # UV
    f3_blue = d1['fxl'] # Blue
    
    ratio = f3_blue / f1_uv if f1_uv > 0 else 0
    return f1_uv, ratio

def calibrate_system(samples=8):
    print("\n[!] CALIBRATING BASELINE...")
    ratios = []
    for _ in range(samples):
        _, r = get_spectral_data()
        ratios.append(r)
        time.sleep(0.1)
    avg = sum(ratios) / len(ratios)
    print(f"[+] Baseline Ratio: {avg:.3f}\n")
    return avg

def main():
    base_ratio = calibrate_system()
    
    print("SYSTEM READY")
    print("Press Enter to scan object...")

    try:
        while True:
            cmd = input("\nReady? ").lower()
            if cmd == 'q': break
            if cmd == 'c': base_ratio = calibrate_system(); continue
            
            uv, ratio = get_spectral_data()
            diff = ratio - base_ratio
            
            # --- RANGE-BASED LOGIC ---
            # If the difference is tiny (between -0.02 and +0.02), it's Plastic/Air
            if abs(diff) < SAFE_ZONE_MARGIN:
                result = "PLASTIC (Baseline/Air match)"
            elif diff < -SAFE_ZONE_MARGIN:
                result = "GLASS (Significant Drop)"
            else:
                # This covers if the ratio spikes high unexpectedly
                result = "UNCERTAIN - Re-scan"

            print(f"{'='*35}")
            print(f"DEBUG | UV: {uv:.0f} | Ratio: {ratio:.3f} | Diff: {diff:+.3f}")
            print(f"FINAL RESULT: {result}")
            print(f"{'='*35}")

    except KeyboardInterrupt:
        print("\nExit.")

if __name__ == "__main__":
    main()