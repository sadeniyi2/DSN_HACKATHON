---
title: DSN x BCT LLM Agents
emoji: 🤖
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: "1.45.1"
python_version: "3.11"
app_file: app.py
pinned: false
---

# DSN x BCT — LLM Agent System

An intelligent multi-agent recommendation and review simulation platform powered by Large Language Models, semantic retrieval, and conversational reasoning.

Built for the DSN x BCT AI Challenge by students of Bowen University, Nigeria.

---

## Overview

This project contains two production-style AI agents deployed in a unified Streamlit application:

### Task A — ReviewSimulatorAgent

Predicts:
- the star rating a user would likely give an unseen product
- the review text they would write
- while preserving the user's reviewing style, tone, sentiment, and personality

The system first models the user's reviewing behaviour from historical interactions before generating a grounded review conditioned on the target product.

---

### Task B — RecommendationAgent

Generates personalised product recommendations using:
- semantic retrieval
- conversational refinement
- reranking with grounded reasoning

The agent:
- retrieves candidate products from a FAISS vector index
- infers user intent from purchase/review history
- reranks recommendations using LLM reasoning
- explains recommendations using signals from prior user behaviour

Supports:
- cold-start recommendations
- multi-turn conversational refinement
- explainable recommendation rationales

---

## System Architecture

```text
User Input
     ↓
LLM Reasoning Layer
     ↓
Persona / Intent Modelling
     ↓
Sentence Transformer Embeddings
     ↓
FAISS Semantic Retrieval
     ↓
LLM Grounded Reranking
     ↓
Final Recommendations / Simulated Reviews
```

---

## Technologies Used

| Component | Technology |
|---|---|
| LLM Inference | Llama 3.3 70B via Groq |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Database | FAISS |
| Backend | Python |
| Frontend | Streamlit |
| Dataset | Amazon Reviews 2023 — All_Beauty |
| Deployment | Hugging Face Spaces |

---

## Repository Structure

| File | Description |
|---|---|
| `app.py` | Main Streamlit application |
| `agent.py` | Review simulation agent |
| `recommender.py` | Recommendation pipeline |
| `catalog.parquet` | Product catalog |
| `catalog.faiss` | FAISS vector index |
| `solution.ipynb` | Full experimental notebook |
| `solution_paper.pdf` | Architecture + evaluation report |
| `metrics_task_a.json` | Review simulation metrics |
| `metrics_task_b.json` | Recommendation metrics |
| `naija_showcase.json` | Nigerian-context examples |
| `coldstart_multiturn.json` | Cold-start + conversational demos |

---

## Evaluation Results

### Task A — Review Simulation

| Metric | Score |
|---|---|
| RMSE | 1.18 |
| MAE | 0.93 |
| Exact Rating Match | 38% |
| Within ±1 Star | 78% |
| ROUGE-L | 0.142 |

---

### Task B — Recommendation

| Metric | Score |
|---|---|
| Hit@10 | 0.130 |
| NDCG@10 | 0.061 |

---

## Nigerian Context Localisation

The platform includes a specialised `naija_mode=True` setting that adapts generated reviews into natural Nigerian English with:
- light pidgin
- code-switching
- culturally grounded phrasing
- preserved sentiment calibration

This demonstrates localisation-aware LLM behaviour for African users.

---

## Dataset

This project uses the McAuley Lab Amazon Reviews 2023 dataset (`All_Beauty` category).

Processing pipeline:
- 500 power users sampled
- leave-last-out evaluation
- persona history generation
- semantic indexing over a 5,000-item catalog

---

## Authors

- Adeniyi Oluwademiladeayo Samuel
- Ukegbu Chidera
- Oyebamire Oluwaseun

Department of Computer Science  
Bowen University, Iwo, Nigeria

---

## Deployment

Hosted on Hugging Face Spaces using Streamlit.

Powered by:
- Groq
- Streamlit
- FAISS
- Sentence Transformers
- Hugging Face
