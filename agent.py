import os, json, re, time
from openai import OpenAI

GROQ_BASE  = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.3-70b-versatile"


def format_persona_for_llm(persona: dict) -> str:
    lines = [
        "User behavioural profile:",
        f"- Average rating given : {persona.get('avg_rating','n/a')}/5",
        f"- Rating distribution  : {persona.get('rating_distribution',{})}",
        f"- Typical review length: ~{persona.get('avg_review_length','n/a')} chars",
        f"- Reviews on record    : {persona.get('history_count', len(persona.get('history',[])))}",
        "",
        "Recent reviews (newest first):",
    ]
    for h in persona.get('history', [])[:8]:
        lines.append(f'  [{h.get("date","")}] "{(h.get("item","unknown"))[:80]}" - {h.get("rating",0)} stars')
        if h.get('excerpt'):
            lines.append(f'    excerpt: {h["excerpt"]}')
    return "\n".join(lines)


class ReviewSimulatorAgent:
    def __init__(self, model_name=GROQ_MODEL, naija_mode=False, api_key=None):
        api_key = api_key or os.environ.get('GROQ_API_KEY')
        if not api_key: raise ValueError('GROQ_API_KEY not set')
        self.client = OpenAI(api_key=api_key, base_url=GROQ_BASE)
        self.model_name = model_name
        self.naija_mode = naija_mode

    def _call(self, prompt, retries=4):
        for i in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
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

    def _naija_overlay(self):
        if not self.naija_mode: return ""
        return ("\nIMPORTANT: This user is Nigerian. Use natural Nigerian English with "
                "comfortable code-switching and light pidgin where it fits. Don't caricature.\n")

    def simulate(self, persona, target_item):
        persona_text = format_persona_for_llm(persona)
        item_text = (
            f"Item title : {target_item.get('title','unknown')}\n"
            f"Categories : {target_item.get('categories', [])}\n"
            f"Description: {(target_item.get('description') or '')[:300]}"
        )
        analysis = self._call(
            f"{persona_text}\n\nIn 2-3 sentences, describe this user's reviewing personality."
        )
        gen_prompt = (
            f"{persona_text}\n\nPersonality analysis:\n{analysis}\n\n"
            f"They are reviewing a NEW item:\n{item_text}\n{self._naija_overlay()}\n"
            "Predict (a) the integer star rating (1-5) and (b) the review THEY would write.\n\n"
            "Respond with ONLY this JSON:\n"
            '{"predicted_rating": <int>, "review_text": "<the review>", "reasoning": "<1 sentence>"}'
        )
        raw = self._call(gen_prompt)
        try:
            out = self._extract_json(raw)
            out['predicted_rating'] = int(round(float(out['predicted_rating'])))
            out['analysis'] = analysis
            return out
        except Exception as e:
            return {'predicted_rating': 3, 'review_text': raw[:500],
                    'reasoning': f'parse_error: {e}', 'analysis': analysis}
