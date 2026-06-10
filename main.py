from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import BaseModel

from model import load_model, generate_with_cache


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int=128

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model, app.state.tokenizer = load_model()
    yield
    del app.state.model
    del app.state.tokenizer


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health(request: Request):
    return {
        "status": "ok",
        "model_loaded": hasattr(request.app.state, "model"),
    }

@app.post("/generate")
def generate(body: GenerateRequest, request: Request):
    text = generate_with_cache(
        request.app.state.model,
        request.app.state.tokenizer,
        body.prompt,
        body.max_new_tokens
    )
    return {"text": text}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)
