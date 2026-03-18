import time, sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from as7343 import AS7343

GAIN = 64
INTEGRATION_TIME = 200

# All channels the library actually returns
TARGET_CHANNELS = ['fz', 'fy', 'fxl', 'nir', 'vis_tl', ]
CHANNEL_LABELS  = ['FZ\n~480nm', 'FY\n~555nm', 'FXL\n~600nm',
                   'NIR\n~855nm', 'VIS-TL', ]
CHANNEL_COLORS  = ['#4466FF', '#00CC44', '#FFAA00', '#FF4422', '#888888',]

try:
    sensor = AS7343()
    sensor.set_gain(GAIN)
    sensor.set_integration_time(INTEGRATION_TIME)
except Exception as e:
    print(f"Hardware Error: {e}"); sys.exit()

def get_data():
    try:
        sensor.bank_select(1)
        time.sleep(0.12)
        raw = sensor.get_data()[0]
        return {k: float(raw[k]) for k in TARGET_CHANNELS if k in raw}
    except Exception as e:
        print(f"Read error: {e}")
        return {}

plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(10, 6))

initial = get_data()
values  = [initial.get(ch, 0.0) for ch in TARGET_CHANNELS]
bars    = ax.bar(CHANNEL_LABELS, values, color=CHANNEL_COLORS, edgecolor='black', width=0.6)
vtexts  = [ax.text(b.get_x() + b.get_width()/2, b.get_height(),
           f'{b.get_height():.0f}', ha='center', va='bottom', fontsize=10) for b in bars]

ax.set_ylabel('Raw Sensor Counts')
ax.set_xlabel('Spectral Channel')
ax.set_title('Live Spectrum — AS7343 (All Available Library Channels)')

def animate(frame):
    data = get_data()
    if data:
        for bar, txt, ch in zip(bars, vtexts, TARGET_CHANNELS):
            val = data.get(ch, 0.0)
            bar.set_height(val)
            txt.set_y(val); txt.set_text(f'{val:.0f}')
        mx = max(data.get(ch, 0.0) for ch in TARGET_CHANNELS)
        ax.set_ylim(0, (mx if mx > 0 else 10) * 1.15)
    return bars

ani = animation.FuncAnimation(fig, animate, interval=350, cache_frame_data=False)
plt.tight_layout(); plt.show()