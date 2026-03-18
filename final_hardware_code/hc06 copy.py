from gpiozero import DigitalInputDevice
import time

# Verify this matches your physical wire!
# Physical Pin 36 = BCM 16
# Physical Pin 16 = BCM 23
pin_to_test = 16 

sensor = DigitalInputDevice(pin_to_test, pull_up=True)

print(f"--- LIVE PIN {pin_to_test} MONITOR ---")
print("1 = No Signal | 0 = Signal (Metal)")
try:
    while True:
        print(f"Current Value: {sensor.value}", end="\r")
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nExiting.")