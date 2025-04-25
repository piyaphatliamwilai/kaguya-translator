import cv2
import numpy as np
from PIL import ImageDraw
from ultralytics import YOLO
import utils
import masking
import translator
import argparse

def main(image_path="test.png"):
    # initialization
    net = cv2.dnn.readNet("models/detect_manga.weights", "models/yolo.cfg")
    MODEL_PATH = "best.pt"
    model = YOLO(MODEL_PATH)
    print("[DEBUG] YOLO model loaded")

    img = cv2.imread(image_path)
    h, w, _ = img.shape

    predicted = model.predict(image_path, conf=0.5)
    predicted[0].save("debug/yolo.png")

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
    draw = translator.apply_translation(pil_img, draw, predicted)

    # Convert back to OpenCV format
    inpainted = utils.convert_to_cv2(pil_draw)
    # Save the final image with text
    cv2.imwrite("translated.png", inpainted)
    print("[DEBUG] Text added to bubbles. Output saved as translated.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Translate manga images")
    parser.add_argument("image", type=str, help="Path to the manga image")
    args = parser.parse_args()
    main(args.image)