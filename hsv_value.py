import cv2
import numpy as np

# --- Your correction LUTs ---
_CORRECTION_LUT_B = None
_CORRECTION_LUT_G = None
_CORRECTION_LUT_R = None

def correct_frame(frame):
    global _CORRECTION_LUT_B, _CORRECTION_LUT_G, _CORRECTION_LUT_R
    if _CORRECTION_LUT_B is None:
        _CORRECTION_LUT_B = np.array([min(255, int(i * 0.457)) for i in range(256)], dtype=np.uint8)
        _CORRECTION_LUT_G = np.array([min(255, int(i * 0.85))  for i in range(256)], dtype=np.uint8)
        _CORRECTION_LUT_R = np.array([min(255, int(i * 0.926)) for i in range(256)], dtype=np.uint8)
    b, g, r = cv2.split(frame)
    return cv2.merge((
        cv2.LUT(b, _CORRECTION_LUT_B),
        cv2.LUT(g, _CORRECTION_LUT_G),
        cv2.LUT(r, _CORRECTION_LUT_R),
    ))

# ----------------------------

clicked_info = {"pos": None, "hsv": None}

def sample_hsv(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        corrected = correct_frame(param["frame"])          # ← apply correction
        hsv_frame = cv2.cvtColor(corrected, cv2.COLOR_BGR2HSV)
        h, s, v = hsv_frame[y, x]
        clicked_info["pos"] = (x, y)
        clicked_info["hsv"] = (int(h), int(s), int(v))

cap = cv2.VideoCapture(0)
frame_data = {"frame": None}

cv2.namedWindow("Live HSV Sampler (Corrected)")
cv2.setMouseCallback("Live HSV Sampler (Corrected)", sample_hsv, frame_data)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_data["frame"] = frame.copy()
    corrected = correct_frame(frame)                       # ← display corrected feed

    if clicked_info["pos"] and clicked_info["hsv"]:
        x, y = clicked_info["pos"]
        h, s, v = clicked_info["hsv"]

        swatch_bgr = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0][0]
        label = f"H:{h}  S:{s}  V:{v}"

        cv2.drawMarker(corrected, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        cv2.rectangle(corrected, (x + 10, y - 40), (x + 220, y + 10), (0, 0, 0), -1)
        cv2.rectangle(corrected, (x + 10, y - 38), (x + 50, y + 8), swatch_bgr.tolist(), -1)
        cv2.putText(corrected, label, (x + 55, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    cv2.putText(corrected, "Click to sample HSV (corrected) | Q to quit", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Live HSV Sampler (Corrected)", corrected)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()