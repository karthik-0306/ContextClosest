import os
import json
from google import genai
from google.genai import types
from groq import Groq

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

class QueryParser:
    def __init__(self, gemini_model_name: str = "gemini-2.0-flash", groq_text_model: str = "llama-3.3-70b-versatile"):
        self.gemini_model_name = gemini_model_name
        self.groq_text_model = groq_text_model

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
            self.groq_client = Groq(api_key=self._groq_keys[0])
        else:
            self.groq_client = None

        self.system_prompt = (
            "You are a fashion search parser. Parse the user's natural language query into structured attributes. "
            "Return JSON only, no markdown: "
            '{"garments": [{"item": "<garment type>", "color": "<color>"}], "setting": "<environment/context description>"}'
        )

    def parse_query(self, query: str) -> dict:
        prompt = f"{self.system_prompt}\n\nQuery: {query}"

        # Try Gemini
        if self.gemini_client:
            try:
                response = self.gemini_client.models.generate_content(
                    model=self.gemini_model_name,
                    contents=prompt
                )
                return parse_json_response(response.text)
            except Exception as e:
                if "limit: 0" in str(e):
                    print("Gemini quota exhausted. Falling back to Groq for query parsing.")
                    self.gemini_client = None
                else:
                    print(f"Gemini query parse failed: {e}. Falling back to Groq...")

        # Try Groq (text-only, much cheaper than vision)
        if self.groq_client:
            try:
                response = self.groq_client.chat.completions.create(
                    model=self.groq_text_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=150
                )
                return parse_json_response(response.choices[0].message.content)
            except Exception as e:
                print(f"Groq query parse failed: {e}.")

        print("Warning: Could not parse query. Returning empty tags.")
        return {"garments": [], "setting": "unknown"}
