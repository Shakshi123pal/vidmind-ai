import requests
import json

body = {
    "video_id": "yt_Lwbr0tFNSmE",
    "question": "what is this video about?",
    "top_k": 5
}

r = requests.post('http://127.0.0.1:8000/ask', json=body)
print('STATUS', r.status_code)
try:
    print(json.dumps(r.json(), indent=2))
except Exception:
    print(r.text)
