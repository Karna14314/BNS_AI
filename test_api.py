from openai import OpenAI
import sys

API_KEY = "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"
BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "meta/llama-3.3-70b-instruct"

def test_api():
    print(f"Testing API with model: {MODEL} (MINIMAL PARAMS)")
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"user","content":"Hi"}],
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024
        )
        print("API Response received!")
        print(f"Content: {completion.choices[0].message.content}")
    except Exception as e:
        print(f"API Error: {e}")

if __name__ == "__main__":
    test_api()
