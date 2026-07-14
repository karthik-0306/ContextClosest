import os
import time
import json
import base64
from io import BytesIO
from google import genai
from google.genai import types
from groq import Groq
from PIL import Image

def _resize_for_api(img: Image.Image, max_size: int = 512) -> Image.Image:
    """Resize image to reduce API token consumption. Keeps aspect ratio."""
    small = img.copy()
    small.thumbnail((max_size, max_size), Image.LANCZOS)
    return small

def encode_image_bytes(img: Image.Image) -> bytes:
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    return buffered.getvalue()

def parse_json_response(response_text: str) -> dict:
    response_text = response_text.strip()
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.startswith("```"):
        response_text = response_text[3:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        print(f"Failed to parse JSON. Raw output: {response_text}")
        return {"garments": [], "setting": "unknown"}

_PROMPT = (
    'List garments and setting. JSON only: '
    '{"garments":[{"item":"<type>","color":"<color>"}],"setting":"<place>"}'
)

class AttributeTagger:
    def __init__(self, gemini_model_name: str = "gemini-2.0-flash", groq_vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"):
        self.gemini_model_name = gemini_model_name
        self.groq_vision_model = groq_vision_model

        # Gemini setup
        genai_key = os.getenv("GEMINI_API_KEY")
        if genai_key:
            self.gemini_client = genai.Client(
                api_key=genai_key,
                http_options=types.HttpOptions(
                    retry_options=types.HttpRetryOptions(attempts=0)
                )
            )
        else:
            self.gemini_client = None

        # Load all Groq keys for rotation
        self._groq_keys = [
            os.getenv("GROQ_API_KEY_1"),
            os.getenv("GROQ_API_KEY_2"),
            os.getenv("GROQ_API_KEY_3"),
        ]
        self._groq_keys = [k for k in self._groq_keys if k]
        self._groq_key_idx = 0

        if self._groq_keys:
            print(f"Loaded {len(self._groq_keys)} Groq API key(s).")
            self._make_groq_client()
        else:
            self.groq_client = None

    def _make_groq_client(self):
        self.groq_client = Groq(api_key=self._groq_keys[self._groq_key_idx])

    def _rotate_groq_key(self) -> bool:
        """Switch to the next available Groq key. Returns False if all are exhausted."""
        self._groq_key_idx += 1
        if self._groq_key_idx < len(self._groq_keys):
            print(f"Switching to Groq key #{self._groq_key_idx + 1}...")
            self._make_groq_client()
            return True
        print("All Groq keys exhausted.")
        self.groq_client = None
        return False

    def _call_gemini(self, img: Image.Image) -> dict:
        small = _resize_for_api(img)
        img_bytes = encode_image_bytes(small)
        try:
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model_name,
                contents=[_PROMPT, types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")]
            )
            return parse_json_response(response.text)
        except Exception as e:
            if "limit: 0" in str(e):
                print("Gemini quota is 0. Disabling Gemini for this run.")
                self.gemini_client = None
            raise

    def _call_groq(self, img: Image.Image) -> dict:
        small = _resize_for_api(img)
        b64 = base64.b64encode(encode_image_bytes(small)).decode("utf-8")
        response = self.groq_client.chat.completions.create(
            model=self.groq_vision_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": _PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]
            }],
            temperature=0.0,
            max_tokens=150
        )
        return parse_json_response(response.choices[0].message.content)

    def tag_image(self, img: Image.Image) -> dict:
        # Try Gemini first
        if self.gemini_client:
            try:
                return self._call_gemini(img)
            except Exception as e:
                print(f"Gemini failed: {e}. Falling back to Groq...")

        # Try Groq with automatic key rotation on rate limit
        while self.groq_client:
            try:
                return self._call_groq(img)
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate_limit_exceeded" in error_str:
                    if "tokens per day" in error_str or "TPD" in error_str:
                        # Daily limit hit — rotate to next key
                        print(f"Groq key #{self._groq_key_idx + 1} daily limit reached.")
                        if not self._rotate_groq_key():
                            break
                    else:
                        # Per-minute rate limit — just wait
                        import re
                        match = re.search(r"try again in ([\d.]+)s", error_str)
                        wait = float(match.group(1)) + 2 if match else 10
                        print(f"Groq per-minute limit. Waiting {wait:.0f}s...")
                        time.sleep(wait)
                else:
                    print(f"Groq API error: {e}.")
                    break

        return {"garments": [], "setting": "unknown"}
