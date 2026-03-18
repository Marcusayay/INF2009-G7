from gpiozero import Button
from signal import pause
import sys

# --- CONFIGURATION ---
# CHANGE THIS if you are using Physical Pin 16. (Physical 16 = BCM 23)
# If your wire is on Physical Pin 36, keep it as 16.
SENSOR_PIN = 16 

def metal_detected():
    print("\n-------------------------")
    print("✅ METAL DETECTED!")
    print("Inductive sensor active.")
    print("-------------------------")

def sensor_cleared():
    print("❌ Sensor Clear.")

def main():
    try:
        # bounce_time=0.05 ignores tiny electrical flickers (noise)
        sensor = Button(SENSOR_PIN, pull_up=True, bounce_time=0.05)
        
        # Link the events
        sensor.when_pressed = sensor_cleared
        sensor.when_released = metal_detected

        print(f"--- Smart Bin System Active ---")
        print(f"Monitoring BCM GPIO: {SENSOR_PIN}")
        print(f"Physical Pin should be: {'36' if SENSOR_PIN == 16 else '??'}")
        print("Waiting for metal detection... (Press Ctrl+C to exit)")

        pause()
        
    except Exception as e:
        print(f"Error starting the sensor: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nShutting down monitor.")

if __name__ == "__main__":
    main()