import os, pickle
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd, numpy as np, faiss
from sentence_transformers import SentenceTransformer

from agent import ReviewSimulatorAgent
from recommender import RecommendationAgent

# ---------- Startup ----------
print("Loading catalog and FAISS index...")
CATALOG  = pd.read_parquet(os.path.join(os.path.dirname(__file__), 'data', 'catalog.parquet'))
FAISS_IX = faiss.read_index(os.path.join(os.path.dirname(__file__), 'data', 'catalog.faiss'))
EMBEDDER = SentenceTransformer('all-MiniLM-L6-v2')
REC_AGENT = RecommendationAgent(CATALOG, FAISS_IX, EMBEDDER)
print(f"ready: {len(CATALOG)} items, {FAISS_IX.ntotal} embeddings")


# ---------- Schemas ----------
class HistoryItem(BaseModel):
    item: str
    rating: int
    excerpt: Optional[str] = ""
    date: Optional[str] = ""

class Persona(BaseModel):
    user_id: Optional[str] = None
    avg_rating: float = Field(3.5, ge=1.0, le=5.0)
    rating_distribution: Dict[str, float] = {}
    avg_review_length: int = 200
    history_count: Optional[int] = None
    history: List[HistoryItem] = []

class TargetItem(BaseModel):
    title: str
    categories: List[str] = []
    description: Optional[str] = ""

# Task A
class SimulateRequest(BaseModel):
    persona: Persona
    target_item: TargetItem
    naija_mode: bool = False

class SimulateResponse(BaseModel):
    predicted_rating: int
    review_text: str
    reasoning: str
    analysis: Optional[str] = None

# Task B
class RecommendRequest(BaseModel):
    persona: Persona
    exclude_asins: List[str] = []
    conversation_context: Optional[str] = None
    top_k: int = 10

class Recommendation(BaseModel):
    parent_asin: str
    rank: int
    reason: str
    title: Optional[str] = None
    categories: Optional[List[str]] = None

class RecommendResponse(BaseModel):
    query: str
    recommendations: List[Recommendation]


# ---------- App ----------
app = FastAPI(title="DSN x BCT - Combined Agent Service", version="1.0.0")

@app.get("/")
def root():
    return {"service": "dsn-bct-agents",
            "tasks": ["A: /simulate-review", "B: /recommend"],
            "status": "ok"}

@app.get("/health")
def health():
    return {"status": "ok",
            "catalog_size": len(CATALOG),
            "faiss_vectors": FAISS_IX.ntotal}

@app.post("/simulate-review", response_model=SimulateResponse)
def simulate_review(req: SimulateRequest):
    try:
        agent = ReviewSimulatorAgent(naija_mode=req.naija_mode)
        return agent.simulate(req.persona.model_dump(), req.target_item.model_dump())
    except ValueError as e:
        raise HTTPException(500, str(e))
    except Exception as e:
        raise HTTPException(500, f"agent_error: {e}")

@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    try:
        result = REC_AGENT.recommend(
            req.persona.model_dump(),
            exclude_asins=set(req.exclude_asins) if req.exclude_asins else None,
            conversation_context=req.conversation_context,
            top_k=req.top_k,
        )
        return result
    except ValueError as e:
        raise HTTPException(500, str(e))
    except Exception as e:
        raise HTTPException(500, f"agent_error: {e}")
