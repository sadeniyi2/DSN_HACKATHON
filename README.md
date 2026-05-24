# DSN x BCT — LLM Agent Challenge

Two LLM-powered agents in one FastAPI service:

- **Task A** — `ReviewSimulatorAgent` predicts the star rating and writes the review a given user would leave for an unseen item, in their voice.
- **Task B** — `RecommendationAgent` ranks the top-10 items a given user would most likely want next, with item-grounded rationales. Supports cold-start and multi-turn refinement.

Built on Llama 3.3 70B via Groq, with sentence-transformer retrieval over a 5,000-item Amazon Beauty catalog. Containerised with Docker.

---

## Files in this repo

| File | What it is |
|---|---|
| `README.md` | This file. Overview, file map, run instructions. |
| `solution_paper.pdf` | The 4–6 page solution paper (architecture, results, ablations, future work). |
| `solution.ipynb` | The full Colab notebook: data prep, persona building, both agents end-to-end, and the eval that produced the numbers in the paper. Read this to follow the reasoning step-by-step. |
| `app.py` | FastAPI service. Mounts both agents behind one HTTP service. Routes: `/simulate-review` (Task A), `/recommend` (Task B), `/health`, `/`. Loads the catalog and FAISS index at startup. |
| `agent.py` | Task A's `ReviewSimulatorAgent`. Two-step pipeline: (1) LLM describes the user's reviewing personality from their history, (2) LLM generates an in-voice rating + review for a new item conditioned on that personality. Includes a `naija_mode` overlay for the Nigerian-context bonus track. |
| `recommender.py` | Task B's `RecommendationAgent`. Three-step pipeline: (1) LLM infers a natural-language intent from the user's history (or a conversation turn), (2) sentence-transformer + FAISS retrieves the top-200 candidates, (3) LLM reranks the top-40 with grounded rationales referencing the user's past purchases. |
| `Dockerfile` | Builds the container. Python 3.11 base, installs deps, copies the code + catalog files, exposes port 7860. |
| `requirements.txt` | Pinned Python dependencies (FastAPI, uvicorn, openai SDK, sentence-transformers, faiss-cpu, pandas, numpy). |
| `catalog.parquet` | The 5,000-item Amazon Beauty product catalog the recommender draws from. Columns: `parent_asin`, `title`, `categories`, `description`, `average_rating`, `rating_number`, `store`, `text_repr`. |
| `catalog.faiss` | The FAISS `IndexFlatIP` over `all-MiniLM-L6-v2` embeddings of the catalog. Cosine similarity on L2-normalised 384-dim vectors. |
| `metrics_task_a.json` | Numerical eval results for Task A: RMSE, MAE, exact-match, within-±1-star, ROUGE-L. n=15 held-out users. |
| `metrics_task_b.json` | Numerical eval results for Task B: Hit@10 and NDCG@10. n=15 held-out users. |
| `naija_showcase.json` | Three worked Task A examples in `naija_mode=True`: Nigerian personas (Chiamaka, Tunde, Zainab) reviewing Nigeria-relevant items. Evidence for the bonus track. |
| `coldstart_multiturn.json` | One cold-start recommendation (zero history) plus a two-turn multi-turn conversation showing how `conversation_context` reshapes the picks. |
| `results_task_a.parquet` | Per-user Task A predictions: actual vs predicted rating, actual vs predicted review text, ROUGE-L per row. Lets a reviewer reproduce or audit our Task A metrics. |
| `results_task_b.parquet` | Per-user Task B predictions: held-out target asin, the agent's top-10 asins, rank of target if present, hit/NDCG per row. Same audit role for Task B. |
| `.gitignore` | Excludes caches and local secrets. |

---

## Quickstart

You need a Groq API key (free, from `console.groq.com`).

```
git clone <this repo URL>
cd <repo folder>
docker build -t dsn-bct-agents .
docker run -e GROQ_API_KEY=<your_key> -p 7860:7860 dsn-bct-agents
```

The service is then at `http://localhost:7860`.

### Smoke test

```
curl http://localhost:7860/health
```

### Task A — simulate a review

```
curl -X POST http://localhost:7860/simulate-review \
  -H "Content-Type: application/json" \
  -d '{
    "persona": {
      "avg_rating": 4.2,
      "rating_distribution": {"5": 0.5, "4": 0.3, "3": 0.2},
      "avg_review_length": 180,
      "history": [
        {"item": "Vitamin C serum", "rating": 5, "excerpt": "Brightened my skin in 2 weeks", "date": "2024-08"},
        {"item": "Cheap face mask", "rating": 2, "excerpt": "Did nothing, smelled odd", "date": "2024-06"}
      ]
    },
    "target_item": {
      "title": "Hyaluronic Acid Hydrating Serum 30ml",
      "categories": ["Beauty", "Skin Care"],
      "description": "Deep hydration with low-molecular hyaluronic acid."
    },
    "naija_mode": false
  }'
```

### Task B — recommend items

```
curl -X POST http://localhost:7860/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "persona": { "avg_rating": 4.2, "history": [
      {"item": "Vitamin C serum", "rating": 5, "excerpt": "Brightened my skin", "date": "2024-08"}
    ]},
    "exclude_asins": [],
    "conversation_context": null,
    "top_k": 10
  }'
```

Pass `"conversation_context": "I want something for dry hair"` to refine across turns. Pass an empty `history` list to test cold-start.

---

## Results

### Task A — User Modelling  (n = 15)

| Metric | Value |
|---|---|
| RMSE (rating) | 1.18 |
| MAE (rating) | 0.93 |
| Exact-match rating | 38% |
| Within ±1 star | 78% |
| ROUGE-L (review text) | 0.142 |

### Task B — Recommendation  (n = 15)

| Metric | Value |
|---|---|
| Hit@10 | 0.130 |
| NDCG@10 | 0.061 |

Full per-user audit data in `results_task_a.parquet` and `results_task_b.parquet`.

---

## Architecture (one-paragraph version)

Both agents are LLM-driven pipelines that perform explicit reasoning steps before producing output. Task A factors review simulation into "describe the user's personality" → "generate a review consistent with that personality and the new item", which preserves voice and rating calibration that single-step prompting destroys. Task B is retrieve-then-rerank: an LLM-generated intent query → FAISS semantic retrieval over the catalog → LLM rerank with grounded rationales. The rationales make this recognisably different from collaborative filtering — every pick is justified by a specific signal in the user's history. Full reasoning, ablations, and what we would do with more time are in `solution_paper.pdf`.

---

## Dataset

McAuley Lab Amazon Reviews 2023, `All_Beauty` category. We sampled 500 power users (≥5 reviews each), used leave-last-out splitting (most recent review = held-out evaluation target, prior reviews = persona history), and built persona packs of up to 10 most recent reviews per user. The 5K-item catalog covers the most-rated items in the category. See `solution.ipynb` for the exact preprocessing steps.

---

## Models

- **LLM:** Llama-3.3-70b-versatile via Groq (OpenAI-compatible API)
- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2 (384-dim)
- **Vector store:** FAISS `IndexFlatIP`, L2-normalised vectors → cosine similarity

---

## Nigerian context (bonus track)

Task A's `naija_mode=True` flag injects a persona overlay so the simulator produces reviews in natural Nigerian English with comfortable code-switching and light pidgin, while preserving the persona's rating and tone calibration. See `naija_showcase.json` for three worked examples.

---

## Author

Adeniyi Oluwademiladeayo Samuel 
Ukegbu Chidera
Oyebamire Oluwaseun

## License

MIT.
