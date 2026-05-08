# test_cameras.py
import cv2

print("🔍 Buscando cámaras disponibles...\n")

for i in range(10):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✅ Cámara {i}: FUNCIONA - {frame.shape[1]}x{frame.shape[0]}")
        else:
            print(f"⚠️ Cámara {i}: Abierta pero no lee frames")
        cap.release()
    else:
        print(f"❌ Cámara {i}: No disponible")

# También probar con URL de DroidCam
print("\n📱 Probando DroidCam HTTP...")
droidcam_urls = [
    "http://192.168.1.100:4747/video",
    "http://192.168.0.100:4747/video",
    "http://10.0.0.1:4747/video",
]

for url in droidcam_urls:
    cap = cv2.VideoCapture(url)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✅ DroidCam en {url}: FUNCIONA")
        cap.release()