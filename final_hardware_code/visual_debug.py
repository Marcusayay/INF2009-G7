import time
import sys
import smbus2
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ═══════════════════════════════════════════════════════════════════════════
# AS7343 — Full 12-Channel Spectrum via Direct I2C (smbus2)
#
# The sensor has 14 photodiodes but only 6 ADCs.
# Two SMUX passes are required to read all 12 spectral channels:
#   Pass A → F1  F2  FZ  F3  F4  FY   (violet → green)
#   Pass B → F5  FXL F6  F7  F8  NIR  (amber  → NIR)
# ═══════════════════════════════════════════════════════════════════════════

I2C_BUS  = 1
I2C_ADDR = 0x39

# ── Register map ─────────────────────────────────────────────────────────────
R_ENABLE  = 0x80   # bit0=PON  bit1=SP_EN
R_ATIME   = 0x81
R_ASTEP_L = 0xCA
R_ASTEP_H = 0xCB
R_CFG1    = 0xAA   # AGAIN (gain)
R_CFG6    = 0xAF   # SMUX_CMD in bits[4:3]
R_STATUS2 = 0xA3   # bit6 = AVALID (data ready)
R_CH0_L   = 0x95   # first of 12 data bytes  (6 × uint16, little-endian)

# ── Sensor settings ──────────────────────────────────────────────────────────
# AGAIN: 0=0.5× 1=1× 2=2× 3=4× 4=8× 5=16× 6=32× 7=64× 8=128× 9=256× 10=512×
AGAIN  = 7    # 64×
ATIME  = 29   # integration time = (ATIME+1)×(ASTEP+1)×2.78µs
ASTEP  = 599  #                  = 30 × 600 × 2.78µs ≈ 50 ms

# ── SMUX Configurations (20 bytes each, written to shadow RAM 0x00–0x13) ─────
# Each nibble routes one photodiode to one of the 6 ADCs.
# ADC order in the array determines which channel appears on CH0–CH5.

# Pass A  →  CH0=F1  CH1=F2  CH2=FZ  CH3=F3  CH4=F4  CH5=FY
SMUX_A = [
    0x00, 0x30, 0x01, 0x00, 0x00, 0x44,
    0x30, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x20, 0x04, 0x00, 0x30, 0x01, 0x50,
    0x00, 0x00
]
PASS_A = [('F1', 415), ('F2', 445), ('FZ', 480),
          ('F3', 515), ('F4', 555), ('FY', 555)]

# Pass B  →  CH0=F5  CH1=FXL  CH2=F6  CH3=F7  CH4=F8  CH5=NIR
SMUX_B = [
    0x00, 0x00, 0x00, 0x40, 0x02, 0x00,
    0x10, 0x03, 0x50, 0x10, 0x03, 0x00,
    0x00, 0x00, 0x24, 0x00, 0x00, 0x00,
    0x50, 0x00
]
PASS_B = [('F5', 590), ('FXL', 630), ('F6', 680),
          ('F7', 720), ('F8',  745), ('NIR', 855)]

ALL_PASS   = PASS_A + PASS_B
ALL_KEYS   = [p[0] for p in ALL_PASS]
ALL_LABELS = [f'{k}\n{nm}nm' for k, nm in ALL_PASS]
ALL_COLORS = [
    '#9B00FF', '#6600CC', '#0044FF',   # F1   F2   FZ   (violet → blue)
    '#00BB44', '#88CC00', '#CCCC00',   # F3   F4   FY   (green → yellow)
    '#FFAA00', '#FF6600', '#FF2200',   # F5   FXL  F6   (amber → red)
    '#CC0055', '#880088', '#5500AA',   # F7   F8   NIR  (deep red → NIR)
]

# ── smbus2 helpers ────────────────────────────────────────────────────────────
try:
    bus = smbus2.SMBus(I2C_BUS)
except Exception as e:
    print(f"Failed to open I2C bus: {e}")
    sys.exit(1)

def rd(reg):      return bus.read_byte_data(I2C_ADDR, reg)
def wr(reg, val): bus.write_byte_data(I2C_ADDR, reg, int(val) & 0xFF)

# ── Sensor init ───────────────────────────────────────────────────────────────
def init_sensor():
    wr(R_ENABLE, 0x01);  time.sleep(0.02)   # power on
    wr(R_ATIME,   ATIME)
    wr(R_ASTEP_L, ASTEP & 0xFF)
    wr(R_ASTEP_H, (ASTEP >> 8) & 0xFF)
    wr(R_CFG1,    AGAIN)
    print("AS7343 initialised  (gain=64×, integration≈50ms)")

# ── SMUX diagnostic (call this if a pass returns all zeros) ───────────────────
def smux_diagnostic():
    print("\n─── SMUX shadow RAM dump (0x00–0x13) ───")
    for i in range(20):
        print(f"  reg 0x{i:02X} = 0x{rd(i):02X}")
    print()

# ── Single measurement pass ───────────────────────────────────────────────────
def measure_pass(smux_bytes):
    """Program SMUX, trigger one integration cycle, return 6 uint16 counts."""
    wr(R_ENABLE, 0x01)                        # disable SP_EN (stop engine)
    time.sleep(0.003)

    for i, b in enumerate(smux_bytes):        # write 20-byte SMUX config
        wr(i, b)                              #   to shadow RAM 0x00-0x13

    wr(R_CFG6,   0x08)                        # SMUX_CMD=01 → commit shadow→internal
    wr(R_ENABLE, 0x03)                        # PON + SP_EN → start integration

    for _ in range(400):                      # poll AVALID (bit6 of STATUS2)
        if rd(R_STATUS2) & 0x40:
            break
        time.sleep(0.003)
    else:
        print("⚠  AVALID timeout — check SMUX config")
        return [0] * 6

    raw = bus.read_i2c_block_data(I2C_ADDR, R_CH0_L, 12)
    return [raw[i * 2] | (raw[i * 2 + 1] << 8) for i in range(6)]

# ── Full 12-channel read ──────────────────────────────────────────────────────
def get_all_channels():
    result = {}
    for smux, pass_info in ((SMUX_A, PASS_A), (SMUX_B, PASS_B)):
        vals = measure_pass(smux)
        for (name, _), val in zip(pass_info, vals):
            result[name] = val
    return result

# ── Startup ───────────────────────────────────────────────────────────────────
try:
    init_sensor()
except Exception as e:
    print(f"Init failed: {e}")
    sys.exit(1)

initial = get_all_channels()
print("First read:", {k: initial[k] for k in ALL_KEYS})

# If all Pass-A or Pass-B values are 0, uncomment the next line to debug:
# smux_diagnostic()

# ── Plot setup ────────────────────────────────────────────────────────────────
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(14, 6))

values = [initial.get(k, 0) for k in ALL_KEYS]
bars   = ax.bar(ALL_LABELS, values, color=ALL_COLORS, edgecolor='black', width=0.7)
vtexts = [
    ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
            f'{b.get_height():.0f}', ha='center', va='bottom', fontsize=8)
    for b in bars
]

# Separator between the two SMUX passes
ax.axvline(5.5, color='gray', linestyle='--', lw=1, alpha=0.6)
ax.text(2.5, 0.98, 'SMUX Pass A', transform=ax.get_xaxis_transform(),
        ha='center', fontsize=9, color='gray')
ax.text(8.5, 0.98, 'SMUX Pass B', transform=ax.get_xaxis_transform(),
        ha='center', fontsize=9, color='gray')

ax.set_ylabel('Raw Counts')
ax.set_xlabel('Spectral Channel')
ax.set_title('AS7343 – Full 12-Channel Spectrum  (Direct I2C / SMUX cycling)')

# ── Animation ─────────────────────────────────────────────────────────────────
def animate(frame):
    data = get_all_channels()
    mx = 10
    for bar, txt, key in zip(bars, vtexts, ALL_KEYS):
        val = data.get(key, 0)
        bar.set_height(val)
        txt.set_y(val)
        txt.set_text(f'{val:.0f}')
        if val > mx:
            mx = val
    ax.set_ylim(0, mx * 1.15)
    return bars

print("\nVisualization running. Close the window to exit.")
ani = animation.FuncAnimation(fig, animate, interval=600, cache_frame_data=False)
plt.tight_layout()
plt.show()

bus.close()