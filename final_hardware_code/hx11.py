import time
import sys
from hx711 import HX711

# 1. Initialize (DT=5, SCK=6)
hx = HX711(dout_pin=15, pd_sck_pin=14)

# --- CALIBRATION SETTINGS ---
# 1. Run the script with nothing on the scale to get your 'OFFSET'
# 2. Put a known weight (e.g., 100g) on, see the 'Raw' value, 
#    and divide it by 100 to get your 'RATIO'
# --- CALIBRATED SETTINGS ---
# --- LOCK IN THESE CALIBRATED SETTINGS ---
RATIO = 492.22  
OFFSET = 0
def get_steady_weight(count=20):
    vals = hx.get_raw_data(count)
    if not vals:
        return 0

    # --- Outlier rejection: drop anything beyond 1.5 × IQR ---
    vals.sort()
    q1 = vals[len(vals) // 4]
    q3 = vals[(3 * len(vals)) // 4]
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    clean = [v for v in vals if lo <= v <= hi]

    if not clean:
        return vals[len(vals) // 2]   # fallback to median

    return sum(clean) / len(clean)    # mean of the clean cluster
def tare():
    global OFFSET
    print("Taring... remove all weights.")
    time.sleep(1)
    OFFSET = get_steady_weight(20)
    print(f"Tare complete. Offset: {OFFSET}")

def main():
    try:
        hx.reset()
        tare()
        
        print("\n--- Starting Measurement ---")
        while True:
            raw_val = get_steady_weight(20)
            
            # The Magic Formula: (Raw Value - Zero Reading) / Calibration Factor
            weight = (raw_val - OFFSET) / RATIO
            
            print(f"Raw: {raw_val:.0f} | Weight: {weight:.2f}g")
            
            time.sleep(0.5)

    except (KeyboardInterrupt, SystemExit):
        print("\nCleaning up...")
        sys.exit()

if __name__ == "__main__":
    main()