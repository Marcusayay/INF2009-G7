import qwiic_as7343
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys

# --- TOGGLE YOUR PLOT HERE ---
# True = Show, False = Hide
show_f1     = True   # UV Baseline
show_f4     = True   # Fluorescence 1
show_f5     = True   # Fluorescence 2 (Greenish)
show_f6     = True   # Fluorescence 3 (Yellowish)
show_clear  = True   # Total Light
show_others = False  # Set to True if you want the NIR/IR stuff

# 1. Initialize Sensor
sensor = qwiic_as7343.QwiicAS7343()
if not sensor.is_connected():
    print("AS7343 not detected!")
    sys.exit()

sensor.begin()
sensor.set_auto_smux(sensor.kAutoSmux18Channels)
sensor.spectral_measurement_enable()

# 2. Setup the Plotting Logic
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 6))

# Map specific names to indices in sensor._data
mapping = {
    "F1 (UV)": 0, "F4 (Cyan)": 3, "F5 (Green)": 4, 
    "F6 (Yellow)": 5, "Clear": 9
}

# Filter based on your booleans above
active_labels = []
active_indices = []

if show_f1: active_labels.append("F1 (UV)"); active_indices.append(0)
if show_f4: active_labels.append("F4 (Cyan)"); active_indices.append(3)
if show_f5: active_labels.append("F5 (Grn)"); active_indices.append(4)
if show_f6: active_labels.append("F6 (Yel)"); active_indices.append(5)
if show_clear: active_labels.append("Clear"); active_indices.append(9)

colors = ["#4B0082", "#00FFFF", "#00FF00", "#FFFF00", "#FFFFFF"]
bars = ax.bar(active_labels, [0]*len(active_labels), color=colors[:len(active_labels)])

ax.set_ylim(0, 8000)
ax.set_title("Plastic vs Glass: Fluorescence Analysis", pad=20)
analysis_text = ax.text(0.5, 0.92, '', transform=ax.transAxes, ha='center', fontsize=12, fontweight='bold')

def update(frame):
    sensor.read_all_spectral_data()
    data = sensor._data
    
    # Update bars
    for i, idx in enumerate(active_indices):
        bars[i].set_height(data[idx])
    
    # MATERIAL LOGIC
    # Comparison: If Green/Yellow (F5/F6) is high relative to UV (F1)
    uv_val = data[0] + 1 # avoid division by zero
    green_val = data[4]
    
    ratio = green_val / uv_val
    
    if uv_val < 50:
        analysis_text.set_text("WAITING FOR UV LIGHT...")
        analysis_text.set_color("white")
    elif ratio > 0.8: # Adjust this threshold after testing your plastic
        analysis_text.set_text(f"RATIO: {ratio:.2f} - LIKELY PLASTIC")
        analysis_text.set_color("#FF5555")
    else:
        analysis_text.set_text(f"RATIO: {ratio:.2f} - LIKELY GLASS")
        analysis_text.set_color("#55FF55")

    return bars

ani = FuncAnimation(fig, update, interval=200, cache_frame_data=False)
plt.show()