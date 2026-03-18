from gpiozero import Button
from signal import pause

# We use 'Button' because a break sensor acts like a switch.
# Pin 26 on the Pi 5.
sensor = Button(26, pull_up=True)

def beam_broken():
    print("🚨 Beam Broken! Object detected.")

def beam_cleared():
    print("✅ Beam Restored.")

# Assign the functions to the sensor events
sensor.when_pressed = beam_broken
sensor.when_released = beam_cleared

print("Monitoring break sensor on Pin 26... Press Ctrl+C to stop.")
pause()