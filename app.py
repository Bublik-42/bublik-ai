
import os

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Bublik AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://bublik-42.github.io",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"


class ChatRequest(BaseModel):
    message: str


@app.get("/")
async def home():
    return {"status": "ok", "service": "Bublik AI"}


@app.post("/chat")
async def chat(request: ChatRequest):
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="На сервере не настроен ключ ИИ.",
        )

    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Напиши сообщение.")

    if len(message) > 8000:
        raise HTTPException(
            status_code=413,
            detail="Сообщение слишком длинное. Максимум 8000 символов.",
        )

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Ты Bublik AI — дружелюбный ИИ-помощник. "
                    "Отвечай по-русски, понятно и по делу. "
                    "Иногда можешь шутить и обращаться к пользователю "
                    "«братан». Не выдумывай факты, если не уверен — скажи."
                ),
            },
            {"role": "user", "content": message},
        ],
        "temperature": 0.7,
        "max_tokens": 1200,
    }

    try:
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="ИИ долго отвечает. Попробуй ещё раз.",
        )
    except httpx.RequestError:
        raise HTTPException(
            status_code=502,
            detail="Не удалось связаться с ИИ.",
        )

    if response.status_code >= 400:
        # Не возвращаем пользователю секретный ключ или полный ответ провайдера.
        raise HTTPException(
            status_code=502,
            detail="Сервис ИИ временно недоступен или достиг лимита запросов.",
        )

    data = response.json()
    try:
        answer = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise HTTPException(
            status_code=502,
            detail="ИИ вернул неожиданный ответ.",
        )

    return {"answer": answer}
