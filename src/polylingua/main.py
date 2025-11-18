from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="PolyLingua", description="A Multilingual Intelligent Speech Assistant")

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)