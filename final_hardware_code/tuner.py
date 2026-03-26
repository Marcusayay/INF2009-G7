"""
AUTO-FIND TAPE HSV — just run it, paste the output to Claude.
No clicking, no interaction. Takes ~5 seconds.
"""
import cv2, numpy as np, time

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
time.sleep(1)  # let camera settle

# grab 5 frames, keep last
for _ in range(5):
    ret, frame = cap.read()
cap.release()

if not ret:
    print("ERROR: camera read failed"); exit()

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
h, w = frame.shape[:2]
print(f"Frame: {w}x{h}")

# --- Dump full H/S/V histograms ---
print("\n=== H HISTOGRAM (S>30, V>30) ===")
mask_sv = (hsv[:,:,1] > 30) & (hsv[:,:,2] > 30)
h_vals = hsv[:,:,0][mask_sv]
for lo in range(0, 180, 5):
    c = ((h_vals >= lo) & (h_vals < lo+5)).sum()
    if c > 0:
        print(f"  H[{lo:3d}-{lo+5:3d}]: {c:5d} {'█'*min(60,c//20)}")

# --- Brute-force scan: try every H-range, find which gives a good lone contour ---
print("\n=== BRUTE FORCE SCAN (looking for tape-sized contour) ===")
best = []
for h_lo in range(0, 170, 5):
    for h_hi in range(h_lo+10, min(h_lo+60, 180), 5):
        for s_lo in [15, 30, 50, 70]:
            for v_lo in [15, 30, 50, 70]:
                m = cv2.inRange(hsv,
                    np.array([h_lo, s_lo, v_lo]),
                    np.array([h_hi, 255, 255]))
                conts, _ = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if not conts:
                    continue
                areas = sorted([cv2.contourArea(c) for c in conts], reverse=True)
                top = areas[0]
                second = areas[1] if len(areas) > 1 else 0
                # We want: top contour is 50-5000px, and is clearly dominant
                if 50 < top < 5000 and top > second * 2:
                    best.append((top, second, h_lo, h_hi, s_lo, v_lo))

best.sort(key=lambda x: (-x[0], x[1]))  # largest top, smallest second
print(f"Found {len(best)} candidate filters")
seen = set()
count = 0
for top, second, h_lo, h_hi, s_lo, v_lo in best:
    key = (h_lo//10, h_hi//10)
    if key in seen:
        continue
    seen.add(key)
    count += 1
    if count > 15:
        break
    print(f"  [{h_lo:3d},{s_lo:3d},{v_lo:3d}]-[{h_hi:3d},255,255]  "
          f"top={top:.0f}px  2nd={second:.0f}px  ratio={top/(second+1):.1f}")

# --- Also dump the 20 brightest non-grey pixels and their HSV ---
print("\n=== 20 BRIGHTEST SATURATED PIXELS (S>60) ===")
sat_mask = hsv[:,:,1] > 60
coords = np.where(sat_mask)
if len(coords[0]) > 0:
    vals = hsv[sat_mask]
    bgrs = frame[sat_mask]
    brightness = vals[:, 2].astype(int)
    idx = np.argsort(-brightness)[:20]
    for i in idx:
        y, x = coords[0][i], coords[1][i]
        hv, sv, vv = vals[i]
        b, g, r = bgrs[i]
        print(f"  ({x:3d},{y:3d}): H={hv:3d} S={sv:3d} V={vv:3d}  BGR=({b},{g},{r})")

# --- Dump a center-region sample ---
print("\n=== CENTER REGION 120x80 SAMPLE (non-dominant H) ===")
cx, cy = w//2, h//2
roi = hsv[cy-40:cy+40, cx-60:cx+60]
roi_bgr = frame[cy-40:cy+40, cx-60:cx+60]
# find dominant H
h_flat = roi[:,:,0].flatten()
dominant_h = np.bincount(h_flat).argmax()
print(f"  Dominant H in center: {dominant_h}")
# show non-dominant pixels
for y in range(0, roi.shape[0], 4):
    for x in range(0, roi.shape[1], 4):
        hv, sv, vv = roi[y, x]
        if abs(int(hv) - int(dominant_h)) > 15 and sv > 30 and vv > 30:
            b, g, r = roi_bgr[y, x]
            print(f"  ({cx-60+x:3d},{cy-40+y:3d}): H={hv:3d} S={sv:3d} V={vv:3d}  BGR=({b},{g},{r})")

print("\n=== DONE — paste everything above to Claude ===")