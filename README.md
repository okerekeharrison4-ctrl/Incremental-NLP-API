# Intent Clarification API (FastAPI)

A beginner-friendly FastAPI template that detects user intent and asks a clarifying question when confidence is low.

## Features

- `POST /clarify-intent` endpoint for intent detection
- Intent labels: `purchase`, `support`, `inquiry`, `other`
- Confidence score output
- Clarifying question generation if confidence is below threshold
- Optional session tracking with JSON file storage (`sessions.json`)
- Deployment-ready with Uvicorn (works on Render/Railway)

## Project Structure

```bash
.
├── main.py
├── requirements.txt
├── sessions.json
└── README.md
```

## Requirements

- Python 3.10+

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run Locally

```bash
uvicorn main:app --reload
```

API will be available at:

- `http://127.0.0.1:8000`
- Interactive docs: `http://127.0.0.1:8000/docs`

## Endpoints

### 1) Health Check

**GET** `/health`

Response:

```json
{
  "status": "ok"
}
```

### 2) Clarify Intent

**POST** `/clarify-intent`

Request JSON:

```json
{
  "text": "I need help with my order",
  "session_id": "optional-session-id"
}
```

Response JSON:

```json
{
  "detected_intent": "support",
  "confidence": 0.9231,
  "clarifying_question": null,
  "session_id": "optional-session-id"
}
```

If confidence is low, example response:

```json
{
  "detected_intent": "inquiry",
  "confidence": 0.5342,
  "clarifying_question": "To help me route this correctly, do you mean inquiry, support, or purchase?",
  "session_id": "d66c3f07-43f0-43ee-9cd8-a58a94a74f93"
}
```

## Python `requests` Example

```python
import requests

url = "http://127.0.0.1:8000/clarify-intent"
payload = {
    "text": "I want to return something and maybe buy a replacement",
    "session_id": "user-123"
}

resp = requests.post(url, json=payload, timeout=30)
print(resp.status_code)
print(resp.json())
```

## Environment Variables (Optional)

Create a `.env` file (optional):

```env
INTENT_CONFIDENCE_THRESHOLD=0.70
SESSION_STORE_PATH=sessions.json
```

- `INTENT_CONFIDENCE_THRESHOLD`: confidence cutoff for asking clarifying questions
- `SESSION_STORE_PATH`: JSON file path for session history

## Notes for Deployment (Render / Railway)

- Start command:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

- First request may be slower because Hugging Face model downloads on first run.
- For production, use persistent storage or external database instead of local JSON storage.

## Error Handling

- Input validation rejects empty `text`
- Unknown fields in request body are rejected
- Returns HTTP 500 for unexpected model/runtime errors

## License

Template project for educational and starter use.
