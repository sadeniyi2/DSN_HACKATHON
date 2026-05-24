import os, json, re, time
import numpy as np, pandas as pd, faiss
from openai import OpenAI
from sentence_transformers import SentenceTransformer

GROQ_BASE  = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.3-70b-versatile"


class RecommendationAgent:
    def __init__(self, catalog, index, embedder, model_name=GROQ_MODEL, api_key=None):
        api_key = api_key or os.environ.get('GROQ_API_KEY')
        if not api_key: raise ValueError('GROQ_API_KEY not set')
        self.client = OpenAI(api_key=api_key, base_url=GROQ_BASE)
        self.model_name = model_name
        self.catalog = catalog
        self.index = index
        self.embedder = embedder

    def _call(self, prompt, retries=4):
        for i in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                msg = str(e)
                if '429' in msg or 'rate' in msg.lower():
                    time.sleep(15 + i*5)
                elif i == retries - 1:
                    raise
                else:
                    time.sleep(2 ** i)
        raise RuntimeError("LLM call failed after retries")

    def _extract_json(self, text):
        text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip(), flags=re.MULTILINE)
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m: text = m.group(0)
        return json.loads(text)

    def _build_query(self, persona, conversation_context=None):
        history_text = '\n'.join(
            f"- {h['rating']}* '{h['item'][:60]}': {h['excerpt'][:100]}"
            for h in persona.get('history', [])[:8]
        )
        cold_start = len(persona.get('history', [])) == 0
        prompt = (
            (f"User review history:\n{history_text}\n" if not cold_start else "User has no prior history (COLD START).\n")
            + f"Average rating: {persona.get('avg_rating','unknown')}\n"
            + (f"Conversation context: {conversation_context}\n" if conversation_context else "")
            + "\nIn one sentence, describe what kind of product this user is most likely to want next. "
            "Be SPECIFIC about product types, key attributes, and qualities they care about."
        )
        return self._call(prompt)

    def _retrieve(self, query_text, exclude_asins=None, k=200):
        qvec = self.embedder.encode([query_text]).astype('float32')
        faiss.normalize_L2(qvec)
        scores, idx = self.index.search(qvec, k)
        cands = self.catalog.iloc[idx[0]].copy()
        cands['retrieval_score'] = scores[0]
        if exclude_asins:
            cands = cands[~cands['parent_asin'].isin(exclude_asins)]
        return cands

    def _rerank(self, persona, candidates, top_k=10, feed_top=40):
        cand_text = '\n'.join(
            f"{i+1}. [{row['parent_asin']}] {row['title'][:80]} (avg {row['average_rating']}*)"
            for i, (_, row) in enumerate(candidates.head(feed_top).iterrows())
        )
        history_text = '\n'.join(
            f"- {h['rating']}* '{h['item'][:50]}'"
            for h in persona.get('history', [])[:6]
        )
        prompt = (
            f"User profile (avg rating {persona.get('avg_rating','n/a')}):\n{history_text}\n\n"
            f"Candidate items (use these EXACT parent_asin codes):\n{cand_text}\n\n"
            f"Pick the top {top_k} items most likely to delight this user. "
            "Respond with ONLY this JSON:\n"
            '{"recommendations":[{"parent_asin":"<exact code>","rank":1,"reason":"<1 sentence>"}]}'
        )
        raw = self._call(prompt)
        try:
            recs = self._extract_json(raw).get('recommendations', [])
            valid = set(candidates.head(feed_top)['parent_asin'])
            recs = [r for r in recs if r.get('parent_asin') in valid]
            if len(recs) < top_k:
                seen = {r['parent_asin'] for r in recs}
                for _, row in candidates.head(feed_top).iterrows():
                    if row['parent_asin'] not in seen:
                        recs.append({'parent_asin': row['parent_asin'], 'rank': len(recs)+1,
                                     'reason': 'retrieval_fallback'})
                    if len(recs) >= top_k: break
            return recs[:top_k]
        except Exception:
            return [{'parent_asin': row['parent_asin'], 'rank': i+1, 'reason': 'parse_fallback'}
                    for i, (_, row) in enumerate(candidates.head(top_k).iterrows())]

    def recommend(self, persona, exclude_asins=None, conversation_context=None, top_k=10):
        query = self._build_query(persona, conversation_context)
        cands = self._retrieve(query, exclude_asins, k=200)
        recs = self._rerank(persona, cands, top_k=top_k, feed_top=40)
        for r in recs:
            row = self.catalog[self.catalog['parent_asin'] == r['parent_asin']]
            if not row.empty:
                r['title'] = row.iloc[0]['title']
                r['categories'] = row.iloc[0]['categories'] if isinstance(row.iloc[0]['categories'], list) else []
        return {'query': query, 'recommendations': recs}
