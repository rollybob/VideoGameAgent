import os, time, cv2, numpy as np
import pygetwindow as gw
import pyautogui

OUT_DIR = "dataset/images/train"
INTERVAL = 0.5  # seconds between frames
TITLE_KEYWORD = "mGBA"  # window title contains this
UPSCALE = 640  # nearest-neighbor upscale short side to 640

os.makedirs(OUT_DIR, exist_ok=True)

def find_emulator_window(title_keyword="mGBA"):
    wins = gw.getWindowsWithTitle(title_keyword)
    if not wins:
        raise RuntimeError(f"No emulator window found with title containing '{title_keyword}'")
    return wins[0]

def capture_window(win):
    left, top, width, height = win.left, win.top, win.width, win.height
    if width <= 0 or height <= 0:
        raise ValueError("Window size invalid.")
    img = pyautogui.screenshot(region=(left, top, width, height))
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return frame

def nn_upscale(frame, target_short=640):
    h, w = frame.shape[:2]
    s = target_short / min(h, w)
    new_w, new_h = int(round(w*s)), int(round(h*s))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

def main():
    win = find_emulator_window(TITLE_KEYWORD)
    print("Capturing from:", win.title)
    while True:
        frame = capture_window(win)
        frame = nn_upscale(frame, UPSCALE)
        ts = int(time.time() * 1000)
        out = os.path.join(OUT_DIR, f"gba_{ts}.png")
        cv2.imwrite(out, frame)
        print("Saved", out)
        time.sleep(INTERVAL)

if __name__ == "__main__":
    main()