from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, saveImage

net = detectNet("SSD-Mobilenet-v2", threshold=0.5)

image_source = videoSource("/home/nvidia/Desktop/1.jpg")
display = videoOutput("display://0")

img = image_source.Capture()
if img is not None:
    detections = net.Detect(img)

    for detection in detections:
        print(f"-- ClassID: {detection.ClassID}")
        print(f"-- Confidence: {detection.Confidence:.5f}")
        print(f"-- Left: {detection.Left:.4f}")
        print(f"-- Top: {detection.Top:.3f}")
        print(f"-- Right: {detection.Right:.0f}")
        print(f"-- Bottom: {detection.Bottom:.3f}")
        print(f"-- Width: {detection.Width:.3f}")
        print(f"-- Height: {detection.Height:.3f}")
        print(f"-- Area: {detection.Area:.0f}")
        print(f"-- Center: ({detection.Center[0]:.3f}, {detection.Center[1]:.3f})")
        print("-" * 30)

    display.Render(img)
    display.SetStatus("Image Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))

    saveImage("/home/nvidia/Desktop/detection_result.jpg", img)

    import time
    time.sleep(5)