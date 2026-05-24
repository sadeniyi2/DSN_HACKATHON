# DSN x BCT — LLM Agent Challenge

Two-agent system for user modelling (Task A) and personalised recommendation (Task B),
built on Llama 3.3 70B via Groq with sentence-transformer retrieval over Amazon Reviews 2023.

## Architecture

### Task A — Review Simulator Agent
Multi-step reasoning pipeline:
1. Persona analysis (LLM characterises reviewing personality from history)
2. Rating prediction grounded in user's rating distribution
3. Review generation in the user's voice and typical length
Optional `naija_mode` overlay for Nigerian-English personas (bonus track).

### Task B — Recommendation Agent
Retrieve-then-rerank architecture:
1. LLM converts persona + optional conversation context into a retrieval query
2. Semantic search over a 5,000-item catalog (all-MiniLM-L6-v2 + FAISS)
3. LLM reranks top-40 candidates with item-grounded rationales
Handles cold-start (empty history) and multi-turn refinement (conversation_context).

## Results

### Task A — User Modelling (n=15)
| Metric | Value |
|---|---|
| RMSE (rating) | 1.366 |
| MAE (rating)  | 0.667 |
| Exact match   | 66.7% |
| Within ±1 star| 86.7% |
| ROUGE-L avg   | 0.256 |

### Task B — Recommendation (n=15)
| Metric | Value |
|---|---|
| Hit@10  | 0.067 |
| NDCG@10 | 0.022 |

## Run locally with Docker

    cd app
    docker build -t dsn-bct-agents .
    docker run -e GROQ_API_KEY=<your_key> -p 7860:7860 dsn-bct-agents

## API endpoints

### POST /simulate-review  (Task A)
Request body: persona, target_item, naija_mode (bool)
Response: predicted_rating, review_text, reasoning, analysis

### POST /recommend  (Task B)
Request body: persona, exclude_asins (list), conversation_context (str|null), top_k (int)
Response: query, recommendations[{parent_asin, rank, reason, title, categories}]

### GET /health
Returns catalog size and FAISS vector count.

## Repo structure

    .
    |- README.md
    |- notebook/solution.ipynb     (full data prep, evals, walkthrough)
    |- app/                        (deployable service)
    |   |- Dockerfile
    |   |- requirements.txt
    |   |- app.py                  (FastAPI app)
    |   |- agent.py                (Task A: review simulator)
    |   |- recommender.py          (Task B: retrieve-then-rerank)
    |   |- data/
    |       |- catalog.parquet     (5K Amazon Beauty items)
    |       |- catalog.faiss       (FAISS index)
    |- results/                    (eval artifacts cited in the paper)
    |- paper/solution_paper.pdf

## Dataset

Amazon Reviews 2023 (McAuley Lab) — All_Beauty category subset, sampled to 500 power users
with 5+ reviews each. Leave-last-out split for held-out evaluation.

## Models

- LLM: Llama-3.3-70b-versatile via Groq API
- Embeddings: sentence-transformers/all-MiniLM-L6-v2
- Vector store: FAISS (IndexFlatIP, cosine similarity)

## Nigerian context bonus

Task A supports a `naija_mode` flag that injects a persona overlay so the simulator
produces reviews in natural Nigerian English with appropriate code-switching.
See `results/naija_showcase.json` for 3 worked examples.

## Team

Adeniyi Oluwademiladeayo Samuel
Ukegbu Chidera 
Oyebamire Seun

