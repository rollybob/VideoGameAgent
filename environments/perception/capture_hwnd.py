
# Windows client-area capture (prevents desktop/UI in frames)
import time, os, cv2, numpy as np
import win32gui, win32ui, win32con

TITLE_KEYWORD = "mGBA"           # adjust to your emulator window title
OUT_DIR = "dataset/images/train" # change as needed
INTERVAL = 0.5                   # seconds between frames
UPSCALE_SHORT = 640              # nearest-neighbor upscale

os.makedirs(OUT_DIR, exist_ok=True)

def find_hwnd(title_keyword):
    def enum_handler(hwnd, result):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title_keyword.lower() in title.lower():
                result.append(hwnd)
    results = []
    win32gui.EnumWindows(enum_handler, results)
    if not results:
        raise RuntimeError(f"No window found containing '{title_keyword}'")
    return results[0]

def grab_client(hwnd):
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    w, h = right - left, bottom - top

    hwin = win32gui.GetWindowDC(hwnd)
    srcdc = win32ui.CreateDCFromHandle(hwin)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, w, h)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (w, h), srcdc, (0, 0), win32con.SRCCOPY)

    signed = bmp.GetBitmapBits(True)
    img = np.frombuffer(signed, dtype='uint8')
    img.shape = (h, w, 4)
    img = img[:, :, :3]  # drop alpha
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    win32gui.DeleteObject(bmp.GetHandle())
    memdc.DeleteDC()
    srcdc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwin)
    return img

def nn_upscale(img, short_side=640):
    H, W = img.shape[:2]
    s = short_side / min(H, W)
    newW, newH = int(round(W*s)), int(round(H*s))
    return cv2.resize(img, (newW, newH), interpolation=cv2.INTER_NEAREST)

def main():
    hwnd = find_hwnd(TITLE_KEYWORD)
    print("Capturing client area of:", win32gui.GetWindowText(hwnd))
    while True:
        frame = grab_client(hwnd)
        frame = nn_upscale(frame, UPSCALE_SHORT)
        ts = int(time.time()*1000)
        out = os.path.join(OUT_DIR, f"gba_{ts}.png")
        cv2.imwrite(out, frame)
        print("Saved", out)
        time.sleep(INTERVAL)

if __name__ == "__main__":
    main()
