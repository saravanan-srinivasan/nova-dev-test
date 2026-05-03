"""
NOVA.DEV — AI Code Assistant Backend
FastAPI server orchestrating an LLM-powered debugging/explanation/optimization pipeline.
"""

from __future__ import annotations

import json
import os
import time
import uuid
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from dotenv import load_dotenv

from models import (
    AnalyzeRequest,
    AnalyzeResponse,
    FollowUpRequest,
    FollowUpResponse,
    HealthRequest,
    HealthResponse,
    Mode,
    Session,
    Message,
    ComplexityResponse,
)
from prompts import build_system_prompt, build_user_message, MODE_CONFIG
from analysis import compute_health_metrics, estimate_complexity
from storage import SessionStore

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")

session_store = SessionStore(Path(__file__).parent / "data" / "sessions.json")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    session_store.load()
    yield
    session_store.save()


app = FastAPI(
    title="NOVA.DEV API",
    description="AI Code Assistant & Debugging Tool — FastAPI backend orchestrating LLM-powered code intelligence.",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Groq client ────────────────────────────────────────────────────────
async def call_groq(
    system_prompt: str,
    messages: list[dict],
    max_tokens: int = 2200,
) -> dict:
    """Send a request to the Groq API and return the parsed response."""
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY not configured. Add it to backend/.env",
        )

    # Build messages in OpenAI-compatible format
    api_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        api_messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    payload = {
        "model": GROQ_MODEL,
        "messages": api_messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    api_url = "https://api.groq.com/openai/v1/chat/completions"

    async with httpx.AsyncClient(timeout=90.0) as client:
        try:
            r = await client.post(api_url, json=payload, headers=headers)
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="LLM request timed out")
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"LLM request failed: {e}")

    if r.status_code != 200:
        raise HTTPException(
            status_code=r.status_code,
            detail=f"Groq API error: {r.text[:300]}",
        )
    return r.json()


def extract_text(resp: dict) -> str:
    """Pull plain text out of Groq's response (OpenAI format)."""
    try:
        choices = resp.get("choices", [])
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "")
    except (IndexError, KeyError):
        return ""


# ─── Routes ──────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "service": "NOVA.DEV API",
        "version": "3.0.0",
        "status": "online",
        "model": GROQ_MODEL,
        "endpoints": [
            "POST /api/analyze",
            "POST /api/analyze/stream",
            "POST /api/followup",
            "POST /api/health",
            "POST /api/complexity",
            "GET /api/sessions",
            "GET /api/sessions/{id}",
            "DELETE /api/sessions/{id}",
            "GET /api/modes",
        ],
    }


@app.get("/api/modes")
async def get_modes():
    """Return available analysis modes (for the frontend to render mode cards)."""
    return {
        "modes": [
            {
                "id": m_id,
                "label": cfg["label"],
                "description": cfg["description"],
                "needs_error": cfg["needs_error"],
                "accent": cfg["accent"],
            }
            for m_id, cfg in MODE_CONFIG.items()
        ]
    }


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    """Run a complete code-analysis pipeline: build prompts, call LLM, persist session."""
    if not req.code.strip():
        raise HTTPException(status_code=400, detail="Code is required")

    cfg = MODE_CONFIG.get(req.mode.value)
    if not cfg:
        raise HTTPException(status_code=400, detail=f"Unknown mode: {req.mode}")
    if cfg["needs_error"] and not req.error_text.strip():
        raise HTTPException(
            status_code=400, detail=f"Mode '{req.mode.value}' requires an error message"
        )

    system_prompt = build_system_prompt(req.mode.value)
    user_message = build_user_message(req.code, req.language, req.error_text, cfg)

    started = time.time()
    resp = await call_groq(
        system_prompt=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    elapsed = time.time() - started

    text = extract_text(resp)
    usage = resp.get("usage", {})
    tokens = usage.get("total_tokens", 0)

    session_id = req.session_id or f"s_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"
    session = Session(
        id=session_id,
        timestamp=int(time.time() * 1000),
        mode=req.mode,
        code=req.code,
        error_text=req.error_text,
        language=req.language,
        response=text,
        chat_history=[
            Message(role="user", content=user_message),
            Message(role="assistant", content=text),
        ],
        tokens=tokens,
    )
    session_store.put(session)

    return AnalyzeResponse(
        session_id=session_id,
        response=text,
        tokens=tokens,
        elapsed_ms=int(elapsed * 1000),
        model=GROQ_MODEL,  # Fixed: was ANTHROPIC_MODEL
        mode=req.mode,
    )


@app.post("/api/analyze/stream")
async def analyze_stream(req: AnalyzeRequest):
    """
    Streaming endpoint that emits Server-Sent Events for the reasoning pipeline.

    The frontend consumes this as a live trace: each pipeline stage is announced,
    then the LLM response arrives, then a 'done' event closes the stream.
    """
    cfg = MODE_CONFIG.get(req.mode.value)
    if not cfg:
        raise HTTPException(status_code=400, detail=f"Unknown mode: {req.mode}")
    if not req.code.strip():
        raise HTTPException(status_code=400, detail="Code is required")
    if cfg["needs_error"] and not req.error_text.strip():
        raise HTTPException(
            status_code=400, detail=f"Mode '{req.mode.value}' requires an error message"
        )

    pipeline_steps = cfg["pipeline"]

    async def event_generator() -> AsyncGenerator[str, None]:
        def sse(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {json.dumps(data)}\n\n"

        # Stage 1: emit pipeline plan
        yield sse(
            "pipeline_plan",
            {
                "steps": pipeline_steps,
                "mode": req.mode.value,
                "language": req.language,
            },
        )

        # Stage 2: walk through pipeline stages with small delays
        for i, step in enumerate(pipeline_steps):
            await asyncio.sleep(0.4)
            yield sse("pipeline_step", {"index": i, "label": step["label"]})

        # Stage 3: actually call the LLM
        yield sse("llm_call_start", {"model": GROQ_MODEL})

        system_prompt = build_system_prompt(req.mode.value)
        user_message = build_user_message(req.code, req.language, req.error_text, cfg)

        try:
            started = time.time()
            resp = await call_groq(
                system_prompt=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            elapsed = time.time() - started
            text = extract_text(resp)
            usage = resp.get("usage", {})
            tokens = usage.get("total_tokens", 0)
        except HTTPException as e:
            yield sse("error", {"message": e.detail})
            return

        session_id = req.session_id or f"s_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"
        session = Session(
            id=session_id,
            timestamp=int(time.time() * 1000),
            mode=req.mode,
            code=req.code,
            error_text=req.error_text,
            language=req.language,
            response=text,
            chat_history=[
                Message(role="user", content=user_message),
                Message(role="assistant", content=text),
            ],
            tokens=tokens,
        )
        session_store.put(session)

        yield sse(
            "response",
            {
                "session_id": session_id,
                "response": text,
                "tokens": tokens,
                "elapsed_ms": int(elapsed * 1000),
            },
        )
        yield sse("done", {})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/followup", response_model=FollowUpResponse)
async def followup(req: FollowUpRequest):
    """Continue a conversation with the same code context."""
    session = session_store.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    history = [{"role": m.role, "content": m.content} for m in session.chat_history]
    history.append({"role": "user", "content": req.message})

    system_prompt = (
        "You are continuing a code-assistance conversation. The user has additional questions "
        "about the same code. Use markdown formatting (### headers, **bold**, `code`, fenced "
        "code blocks). Be precise and concise."
    )

    started = time.time()
    resp = await call_groq(system_prompt=system_prompt, messages=history)
    elapsed = time.time() - started

    text = extract_text(resp)
    usage = resp.get("usage", {})
    tokens = usage.get("total_tokens", 0)

    session.chat_history.append(Message(role="user", content=req.message))
    session.chat_history.append(Message(role="assistant", content=text))
    session.tokens += tokens
    session.timestamp = int(time.time() * 1000)
    session_store.put(session)

    return FollowUpResponse(
        response=text,
        tokens=tokens,
        elapsed_ms=int(elapsed * 1000),
    )


@app.post("/api/health", response_model=HealthResponse)
async def health(req: HealthRequest):
    """Compute static code-health metrics (no LLM call) — fast and free."""
    metrics = compute_health_metrics(req.code, req.language)
    return HealthResponse(**metrics)


@app.post("/api/complexity", response_model=ComplexityResponse)
async def complexity(req: HealthRequest):
    """Estimate big-O complexity of a function (LLM-powered, fast)."""
    if not req.code.strip():
        raise HTTPException(status_code=400, detail="Code is required")

    system_prompt = (
        "You analyze code complexity. Reply with JSON ONLY (no markdown, no explanation outside JSON):\n"
        '{"time": "O(...)", "space": "O(...)", "explanation": "<short sentence>", "confidence": "high|medium|low"}'
    )
    user_msg = f"Analyze this {req.language} code:\n\n```{req.language}\n{req.code}\n```"

    resp = await call_groq(
        system_prompt=system_prompt,
        messages=[{"role": "user", "content": user_msg}],
        max_tokens=300,
    )
    text = extract_text(resp).strip()
    # strip markdown fences if model wraps anyway
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = estimate_complexity(req.code)
    return ComplexityResponse(**parsed)


@app.get("/api/sessions")
async def list_sessions():
    """Return all saved sessions, newest first."""
    sessions = session_store.list_all()
    sessions.sort(key=lambda s: s.timestamp, reverse=True)
    return {"sessions": [s.model_dump() for s in sessions], "count": len(sessions)}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    s = session_store.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    return s.model_dump()


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    if not session_store.delete(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": session_id}


@app.get("/api/stats")
async def get_stats():
    """Aggregate stats across all sessions — used for the dashboard."""
    sessions = session_store.list_all()
    total_tokens = sum(s.tokens for s in sessions)
    by_mode = {}
    for s in sessions:
        by_mode[s.mode.value] = by_mode.get(s.mode.value, 0) + 1
    by_language = {}
    for s in sessions:
        by_language[s.language] = by_language.get(s.language, 0) + 1
    return {
        "total_sessions": len(sessions),
        "total_tokens": total_tokens,
        "by_mode": by_mode,
        "by_language": by_language,
    }


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {str(exc)}"},
    )


if __name__ == "__main__":
    import uvicorn
    # Port binding for Render deployment (reads PORT env var, defaults to 8000 for local dev)
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
