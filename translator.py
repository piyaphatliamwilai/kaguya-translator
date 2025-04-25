import requests

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