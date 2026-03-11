import qwiic_as7343
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys

# --- CONFIGURATION: TOGGLE CHANNELS HERE ---
# Set to True to plot, False to hide
plot_config = {
    "F1 (405nm)":  True,   # Essential for UV baseline
    "F2 (425nm)":  True,
    "F3 (450nm)":  True,
    "F4 (475nm)":  True,   # Key for Fluorescence
    "F5 (515nm)":  True,   # Key for Fluorescence
    "F6 (555nm)":  True,
    "F7 (600nm)":  False,
    "F8 (670nm)":  False,
    "NIR":         False,
    "Clear":       True,   # Essential for total reflection
    "FD":          False,
    "IR1":         False,
    "IR2":         False,
    "IR3":         False
}

# 1. Initialize Sensor
sensor = qwiic_as7343.QwiicAS7343()
if not sensor.is_connected():
    print("AS7343 not detected!")
    sys.exit()

sensor.begin()
sensor.set_auto_smux(sensor.kAutoSmux18Channels)
sensor.spectral_measurement_enable()

# 2. Filter labels and colors based on config
all_channels = [
    ("F1", "#4B0082"), ("F2", "#0000FF"), ("F3", "#007FFF"), ("F4", "#00FFFF"),
    ("F5", "#00FF00"), ("F6", "#FFFF00"), ("F7", "#FF7F00"), ("F8", "#FF0000"),
    ("NIR", "#8B0000"), ("Clear", "#D3D3D3"), ("FD", "#808080"), 
    ("IR1", "#550000"), ("IR2", "#440000"), ("IR3", "#330000")
]

active_indices = [i for i, (name, _) in enumerate(all_channels) if plot_config.get(list(plot_config.keys())[i])]
labels = [all_channels[i][0] for i in active_indices]
colors = [all_channels[i][1] for i in active_indices]

# 3. Setup Plot
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.bar(labels, [0]*len(labels), color=colors)

ax.set_ylim(0, 5000)
ax.set_title("AS7343 Material Analysis: UV Fluorescence", fontsize=16, pad=20)
text_status = ax.text(0.5, 0.95, '', transform=ax.transAxes, ha='center', fontsize=14, fontweight='bold')

def update(frame):
    sensor.read_all_spectral_data()
    full_data = sensor._data[:14]
    
    # Filter data for active bars
    plot_data = [full_data[i] for i in active_indices]
    
    for bar, val in zip(bars, plot_data):
        bar.set_height(val)

    # --- SIMPLE LOGIC FOR GLASS VS PLASTIC ---
    # If UV (F1) is high, but we see significant Green/Yellow (F5/F6)
    # it indicates fluorescence usually found in plastics.
    f1_val = full_data[0]
    f5_val = full_data[4]
    
    if f1_val > 100: # Threshold to ensure UV light is actually on
        ratio = f5_val / (f1_val + 1)
        if ratio > 0.5:
            text_status.set_text("DETECTED: LIKELY PLASTIC (Fluorescence High)")
            text_status.set_color("#FF5555")
        else:
            text_status.set_text("DETECTED: LIKELY GLASS (Fluorescence Low)")
            text_status.set_color("#55FF55")
    else:
        text_status.set_text("STATUS: UV SOURCE OFF")
        text_status.set_color("white")

    return bars

ani = FuncAnimation(fig, update, interval=150, cache_frame_data=False)
plt.tight_layout()
plt.show()