"""Entrypoint for free cloud Python runners (Render / Hugging Face Gradio SDK)."""
import os
import uvicorn
from app.main import app

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        ws_ping_interval=15,
        ws_ping_timeout=20,
    )
