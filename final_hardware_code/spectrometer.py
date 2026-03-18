import time
from as7343 import AS7343

print("Initializing Sensor...")
sensor = AS7343()

# Turn on the LED and set the safe gain we established
sensor.set_illumination_led(False)
sensor.set_gain(64) 

def calibrate(samples=10):
    print("\n--- CALIBRATION ---")
    print("Ensure the light is ON and the sensor is CLEAR.")
    time.sleep(3)
    
    total = 0
    for i in range(samples):
        # Read the default bank
        readings = sensor.get_data()
        data_dict = readings[0] if isinstance(readings, tuple) else readings
        
        # Lock onto the massive Broadband signal instead of F1
        val = data_dict.get('vis_br') or data_dict.get('VIS_BR')
        
        if i == 0:
            print(f"Successfully locked onto the Broadband channel! First raw reading: {val}")
            
        total += val
        time.sleep(0.1)
        
    baseline = total / samples
    print(f"Calibration complete. Baseline Broadband value: {baseline:.2f}")
    return baseline

def classify_material(baseline, threshold_pct=60.0):
    print("\n--- DETECTION ALGORITHM ---")
    print(f"Using a cutoff threshold of {threshold_pct}%\n")
    
    while True:
        input("Place material between the light and sensor, then press Enter...")
        
        readings = sensor.get_data()
        data_dict = readings[0] if isinstance(readings, tuple) else readings
        current_val = data_dict.get('vis_br') or data_dict.get('VIS_BR')
        
        transmission_pct = (current_val / baseline) * 100
        print(f"Reading: {current_val} | Transmission: {transmission_pct:.1f}%")
        
        # Threshold Logic mapped directly to the physical platform layout
        if transmission_pct < threshold_pct:
            print("Result: BLOCKED (Low Transmission)")
            print("-> Action: Rotate CLOCKWISE to Compartment 1 or 4\n")
            # [INSERT CLOCKWISE MOTOR CODE HERE]
            
        else:
            print("Result: TRANSMITTED (High Transmission)")
            print("-> Action: Rotate ANTICLOCKWISE to Compartment 2 or 3\n")
            # [INSERT ANTICLOCKWISE MOTOR CODE HERE]

if __name__ == "__main__":
    try:
        baseline_value = calibrate()
        
        # Test your specific glass/plastic to dial in this number!
        classify_material(baseline_value, threshold_pct=60.0)
        
    except KeyboardInterrupt:
        print("\nExiting program.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        # Turn off the LED when we quit
        sensor.set_illumination_led(False)