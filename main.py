import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import textwrap
from manga_ocr import MangaOcr
import requests
import time

# initialization
net = cv2.dnn.readNet("models/detect_manga.weights", "models/yolo.cfg")
image_path = "image.png"
manga_ocr = MangaOcr()
MODEL_PATH = "runs/detect/train3/weights/best.pt"
model = YOLO(MODEL_PATH)
print("[DEBUG] YOLO model loaded")

# Translation function using REST API approach
def auto_translate(text, src='ja', dest='en'):
    try:
        if not text or text.strip() == "":
            return ""
            
        # Dictionary of common manga phrases for faster, more reliable translation
        common_phrases = {
            "こんにちは": "Hello",
            "さようなら": "Goodbye",
            "ありがとう": "Thank you",
            "はい": "Yes",
            "いいえ": "No",
            "なに": "What",
            "だれ": "Who",
            "どこ": "Where",
            "どうして": "Why",
            "おはよう": "Good morning",
            "こんばんは": "Good evening",
            "すみません": "Excuse me",
            "ごめんなさい": "I'm sorry",
            "大丈夫": "It's okay",
            "行こう": "Let's go",
            "待って": "Wait",
            "止まれ": "Stop",
            "急いで": "Hurry",
            "わかった": "I understand"
        }
        
        # First check if it's a common phrase
        if text in common_phrases:
            return common_phrases[text]
        
        # If not a common phrase, try using a translation API
        try:
            # Using MyMemory Translation API (free, no authentication required for small usage)
            url = f"https://api.mymemory.translated.net/get?q={text}&langpair={src}|{dest}"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if "responseData" in data and "translatedText" in data["responseData"]:
                    return data["responseData"]["translatedText"]
        except Exception as api_error:
            print(f"[ERROR] API Translation failed: {api_error}")
        
        # If API fails, check if parts of the text match common phrases
        for phrase, translation in common_phrases.items():
            if phrase in text:
                return translation
                
        # Last resort: return the original text
        print(f"[WARNING] All translation methods failed for: {text}")
        return text
        
    except Exception as e:
        print(f"[ERROR] Translation failed: {e}")
        return text  # Return original text if translation fails

# Function to add text to speech bubbles with proper wrapping
def add_text_to_bubble(draw, box, text, font_path="arial.ttf"):
    x, y, w, h = box
    
    # Add padding inside the bubble
    padding = 10
    x += padding
    y += padding
    w -= padding * 2
    h -= padding * 2
    
    # Estimate font size based on bubble height
    # Start with a reasonable size and adjust if needed
    font_size = int(h / 6)  # Initial guess
    min_font_size = 12  # Don't go smaller than this
    
    font = None
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        # Fallback to default
        font = ImageFont.load_default()
        font_size = 12
    
    # Calculate how many characters can fit per line based on width
    avg_char_width = font.getlength("A")  # Approximate width of a character
    chars_per_line = max(1, int(w / avg_char_width))
    
    # Wrap text
    wrapped_text = textwrap.fill(text, width=chars_per_line)
    lines = wrapped_text.split('\n')
    
    # If too many lines, reduce font size and try again
    while len(lines) * font_size > h and font_size > min_font_size:
        font_size -= 2
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()
        avg_char_width = font.getlength("A")
        chars_per_line = max(1, int(w / avg_char_width))
        wrapped_text = textwrap.fill(text, width=chars_per_line)
        lines = wrapped_text.split('\n')
    
    # Calculate vertical centering
    total_text_height = len(lines) * font_size
    start_y = y + (h - total_text_height) // 2
    
    # Draw each line of text
    for i, line in enumerate(lines):
        # Calculate horizontal centering for this line
        line_width = font.getlength(line)
        start_x = x + (w - line_width) // 2
        
        # Draw the text
        draw.text((start_x, start_y + i * font_size), line, fill=(0, 0, 0), font=font)
    
    return draw

img = cv2.imread(image_path)
h, w, _ = img.shape

predicted = model.predict(image_path, conf=0.5)
predicted[0].save("yolo.png")

# preprocessing stuff
blob = cv2.dnn.blobFromImage(img, 0.00392, (512, 512), (0, 0, 0), swapRB=True, crop=False)
net.setInput(blob)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
outs = net.forward(output_layers)

# Postprocess YOLO outputs
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

# === [Text Masking using Adaptive Threshold] ===
mask = np.zeros((h, w), dtype=np.uint8)

for i in indexes.flatten():
    x, y, bw, bh = boxes[i]
    x-=10
    y-=10
    bw+=20
    bh+=20
    crop = img[y:y + bh, x:x + bw]
    if crop.size == 0: continue

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Place this mask back on the full image mask
    roi = mask[y:y + bh, x:x + bw]
    mask[y:y + bh, x:x + bw] = cv2.bitwise_or(roi, binary)

# Save mask and original resized for SD
cv2.imwrite("precise_text_mask.png", mask)
cv2.imwrite("detected_original.png", img)
inpainted = cv2.inpaint(img, mask, 7, cv2.INPAINT_TELEA)
cv2.imwrite("inpainted.png", inpainted)

# Convert OpenCV image to PIL for better text rendering
pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
pil_draw = Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(pil_draw)

# Create a dictionary to store original text and translations
translation_data = {}

# Apply text to each detected box from the YOLO prediction
for idx, result in enumerate(predicted):
    translation_idx = 0
    for box in result.boxes:
        # Get bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        bubble_box = [int(x1), int(y1), int(x2-x1), int(y2-y1)]
        cropped = pil_img.crop((int(x1), int(y1), int(x2), int(y2)))
        
        # Save the cropped bubble for debugging
        cropped.save(f"bubble_{translation_idx}.png")
        
        # Perform OCR on the cropped bubble
        manga_ocr_result = manga_ocr(cropped)
        if manga_ocr_result is None or manga_ocr_result.strip() == "":
            print(f"[DEBUG] No text detected in bubble {translation_idx}.")
            continue
        
        print(f"[OCR] Detected text: {manga_ocr_result}")
        
        # Translate the detected text
        translated_text = auto_translate(manga_ocr_result)
        
        # Store original and translation
        translation_data[translation_idx] = {
            "original": manga_ocr_result,
            "translation": translated_text,
            "position": bubble_box
        }
        
        print(f"[TRANSLATION] {manga_ocr_result} → {translated_text}")
        
        # Add text to the bubble
        draw = add_text_to_bubble(draw, bubble_box, translated_text, font_path="arial.ttf")
        translation_idx += 1
        
        # Add a small delay to prevent API rate limiting if using API
        time.sleep(0.5)

# Save translation data to a text file for reference
with open("translation_data.txt", "w", encoding="utf-8") as f:
    f.write("Bubble ID | Original Text | Translated Text\n")
    f.write("-" * 60 + "\n")
    for idx, data in translation_data.items():
        f.write(f"{idx} | {data['original']} | {data['translation']}\n")

# Convert back to OpenCV format
inpainted = cv2.cvtColor(np.array(pil_draw), cv2.COLOR_RGB2BGR)

# Save the final image with text
cv2.imwrite("translated.png", inpainted)
print(f"[DEBUG] Translation completed. Found {len(translation_data)} text bubbles.")
print("[DEBUG] Text added to bubbles. Output saved as translated.png")
print("[DEBUG] Translation data saved to translation_data.txt")