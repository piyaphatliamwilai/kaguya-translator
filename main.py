import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from manga_ocr import MangaOcr
import utils
import masking
import time
import drawing
import translator

# initialization
net = cv2.dnn.readNet("models/detect_manga.weights", "models/yolo.cfg")
image_path = "image.png"
manga_ocr = MangaOcr()
MODEL_PATH = "runs/detect/train3/weights/best.pt"
model = YOLO(MODEL_PATH)
print("[DEBUG] YOLO model loaded")

img = cv2.imread(image_path)
h, w, _ = img.shape

predicted = model.predict(image_path, conf=0.5)
predicted[0].save("debug/yolo.png")

# preprocessing stuff
blob = cv2.dnn.blobFromImage(img, 0.00392, (512, 512), (0, 0, 0), swapRB=True, crop=False)
net.setInput(blob)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
outs = net.forward(output_layers)

boxes, confidences = [], []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.25:
            center_x, center_y = int(detection[0] * w), int(detection[1] * h)
            bw, bh = int(detection[2] * w), int(detection[3] * h)
            x, y = int(center_x - bw / 2), int(center_y - bh / 2)
            boxes.append([x, y, bw, bh])
            confidences.append(float(confidence))

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.45)
mask = np.zeros((h, w), dtype=np.uint8)

for i in indexes.flatten():
    x, y, bw, bh = boxes[i]
    x, y, bw, bh = utils.apply_padding(x, y, bw, bh, 10)
    crop = img[y:y + bh, x:x + bw]
    if crop.size == 0: continue
    binary = masking.adaptive_threshold(crop)
    roi = mask[y:y + bh, x:x + bw]
    mask[y:y + bh, x:x + bw] = cv2.bitwise_or(roi, binary)

cv2.imwrite("debug/precise_text_mask.png", mask)
cv2.imwrite("debug/detected_original.png", img)
inpainted = cv2.inpaint(img, mask, 7, cv2.INPAINT_TELEA)
cv2.imwrite("debug/inpainted.png", inpainted)

pil_img = utils.convert_to_pil(img)
pil_draw = utils.convert_to_pil(inpainted)
draw = ImageDraw.Draw(pil_draw)

# translate part
for idx, result in enumerate(predicted):
    translation_idx = 0
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        bubble_box = [int(x1), int(y1), int(x2-x1), int(y2-y1)]
        cropped = pil_img.crop((int(x1), int(y1), int(x2), int(y2)))
        cropped.save(f"bubble_{translation_idx}.png")
        manga_ocr_result = manga_ocr(cropped)
        if manga_ocr_result is None or manga_ocr_result.strip() == "":
            print(f"[OCR] No text detected in bubble {translation_idx}. Skipping...")
            continue
        print(f"[OCR] Detected text: {manga_ocr_result}")
        translated_text = translator.translate(manga_ocr_result)
        print(f"[TRANSLATION] {manga_ocr_result} → {translated_text}")
        draw = drawing.add_text_to_bubble(draw, bubble_box, translated_text, font_path="arial.ttf")
        translation_idx += 1
        time.sleep(0.5)

# Convert back to OpenCV format
inpainted = utils.convert_to_cv2(pil_draw)
# Save the final image with text
cv2.imwrite("translated.png", inpainted)
print("[DEBUG] Text added to bubbles. Output saved as translated.png")