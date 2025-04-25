import requests
from PIL import Image
from manga_ocr import MangaOcr
import time
import drawing
import translator
import drawing

manga_ocr = MangaOcr()
# Translation function using REST API approach
def translate(text, src='ja', dest='en'):
    try:
        if not text or text.strip() == "":
            return ""
        
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
                
        # Last resort: return the original text
        print(f"[WARNING] All translation methods failed for: {text}")
        return text # can't use the api
        
    except Exception as e:
        print(f"[ERROR] Translation failed: {e}")
        return text  # translation failed, return original text
    
def apply_translation(original_image, draw, predicted):
    for idx, result in enumerate(predicted):
        translation_idx = 0
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bubble_box = [int(x1), int(y1), int(x2-x1), int(y2-y1)]
            cropped = original_image.crop((int(x1), int(y1), int(x2), int(y2)))
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
    return draw