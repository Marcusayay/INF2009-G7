import time
import sys
from as7343 import AS7343

# Hardware Setup
try:
    sensor = AS7343()
    sensor.set_gain(64)
    sensor.set_integration_time(200)
except Exception as e:
    print(f"Hardware Error: {e}"); sys.exit()

def get_fresh_data():
    """Forces the sensor to take a new measurement and returns (UV, Blue, Ratio)."""
    # Force a stop and start to clear stale data
    sensor.stop_measurement()
    sensor.bank_select(1)
    sensor.start_measurement()
    
    # Wait for integration time (200ms) + buffer
    time.sleep(0.3)
    
    d = sensor.get_data()[0]
    uv = d['fz']
    blue = d['fxl']
    ratio = blue / uv if uv > 0 else 0
    return uv, blue, ratio

def calibrate_material(name):
    print(f"\n>>> STEP: Calibrating {name}")
    input(f"Place {name} in position and press [Enter]...")
    
    uv_samples = []
    ratio_samples = []
    
    for i in range(5):
        uv, blue, r = get_fresh_data()
        uv_samples.append(uv)
        ratio_samples.append(r)
        print(f"  Sample {i+1}: UV={uv:<5} | Ratio={r:.3f}")
        time.sleep(0.1)
    
    avg_uv = sum(uv_samples) / 5
    avg_ratio = sum(ratio_samples) / 5
    return avg_uv, avg_ratio

def main():
    print("=== SENSOR REFRESH TRAINING MODE ===")
    
    # 1. Calibrate
    uv_empty, ratio_empty     = calibrate_material("EMPTY BIN")
    uv_glass, ratio_glass     = calibrate_material("GLASS SAMPLE")
    uv_plastic, ratio_plastic = calibrate_material("PLASTIC SAMPLE")

    # 2. Threshold Logic
    # We use BOTH UV drop and Ratio shift
    ratio_threshold = (ratio_glass + ratio_plastic) / 2
    
    print("\n" + "="*40)
    print(f"TRAINING COMPLETE")
    print(f"Baseline UV: {uv_empty:.0f}")
    print(f"Glass UV: {uv_glass:.0f} | Plastic UV: {uv_plastic:.0f}")
    print(f"Decision Threshold: {ratio_threshold:.3f}")
    print("="*40)

    try:
        while True:
            input("\nPress [Enter] to Scan...")
            uv, blue, r = get_fresh_data()
            
            # Use UV intensity to verify an object is actually there
            if uv > (uv_empty * 0.95):
                print(">>> WARNING: UV too high. Is there an object in the way?")
            
            # Determine Material
            if ratio_glass < ratio_plastic:
                result = "PLASTIC" if r > ratio_threshold else "GLASS"
            else:
                result = "GLASS" if r > ratio_threshold else "PLASTIC"

            print(f"RESULT: {result} (UV: {uv:.0f}, Ratio: {r:.3f})")

    except KeyboardInterrupt:
        print("\nExit.")

if __name__ == "__main__":
    main()