from fastapi import FastAPI
from pydantic import BaseModel
import requests

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allows all (good for dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set")

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are an AI assistant for Krish's portfolio. Answer professionally."},
            {"role": "user", "content": req.message}
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    result = response.json()

    reply = result["choices"][0]["message"]["content"]

    return {"reply": reply}