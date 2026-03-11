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
def get_steady_weight(count=15):
    vals = hx.get_raw_data(count)
    if not vals: return 0
    
    # Sort the numbers and pick the middle one (Median)
    # This automatically throws away the "crazy" high/low spikes
    vals.sort()
    median_val = vals[len(vals) // 2]
    return median_val

def tare():
    global OFFSET
    print("Taring... remove all weights.")
    time.sleep(1)
    OFFSET = get_steady_weight(10)
    print(f"Tare complete. Offset: {OFFSET}")

def main():
    try:
        hx.reset()
        tare()
        
        print("\n--- Starting Measurement ---")
        while True:
            raw_val = get_steady_weight(5)
            
            # The Magic Formula: (Raw Value - Zero Reading) / Calibration Factor
            weight = (raw_val - OFFSET) / RATIO
            
            print(f"Raw: {raw_val:.0f} | Weight: {weight:.2f}g")
            
            time.sleep(0.5)

    except (KeyboardInterrupt, SystemExit):
        print("\nCleaning up...")
        sys.exit()

if __name__ == "__main__":
    main()