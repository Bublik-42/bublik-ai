
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
    allow_methods=["GET", "POST"],
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
            detail="На Render не настроен GROQ_API_KEY."
        )

    message = request.message.strip()

    if not message:
        raise HTTPException(status_code=400, detail="Напиши сообщение.")

    if len(message) > 8000:
        raise HTTPException(
            status_code=413,
            detail="Сообщение слишком длинное. Максимум 8000 символов."
        )

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Ты Bublik AI — дружелюбный ИИ-помощник. "
                    "Отвечай по-русски, понятно и по делу. "
                    "Иногда шути и обращайся к пользователю «братан». "
                    "Не выдумывай факты, если не уверен — скажи."
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
        print("Groq error: request timed out")
        raise HTTPException(
            status_code=504,
            detail="ИИ слишком долго отвечает. Попробуй ещё раз."
        )

    except httpx.RequestError as error:
        print(f"Groq connection error: {type(error).__name__}")
        raise HTTPException(
            status_code=502,
            detail="Сервер не смог связаться с сервисом ИИ."
        )

    if response.status_code >= 400:
        try:
            error_data = response.json().get("error", {})
            error_type = error_data.get("type", "unknown")
            error_code = error_data.get("code", "unknown")
            error_message = error_data.get("message", "Нет описания")
        except (ValueError, AttributeError):
            error_type = "unknown"
            error_code = "unknown"
            error_message = response.text[:300]

        # В логах не печатаем API-ключ.
        print(
            f"Groq API error: status={response.status_code}; "
            f"type={error_type}; code={error_code}; "
            f"message={str(error_message)[:500]}"
        )

        raise HTTPException(
            status_code=502,
            detail=(
                f"Ошибка сервиса ИИ: HTTP {response.status_code}. "
                f"{str(error_message)[:300]}"
            ),
        )

    try:
        data = response.json()
        answer = data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError):
        print("Groq error: unexpected response format")
        raise HTTPException(
            status_code=502,
            detail="ИИ вернул ответ в неизвестном формате."
        )

    return {"answer": answer}
