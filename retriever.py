"""
BM25 + TF-IDF retriever over the SHL catalog.
No external model downloads required.
"""
import json, re, pickle
from pathlib import Path
from typing import List, Dict
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CATALOG_PATH = Path("data/catalog.json")
INDEX_PATH   = Path("data/retriever.pkl")

TEST_TYPE_LABELS = {
    "A": "Ability Aptitude Cognitive Reasoning Numerical Verbal Inductive Deductive Spatial",
    "B": "Biodata Situational Judgement SJT Judgment",
    "C": "Competencies Competency Interview Behavioural Behavioral",
    "D": "Development 360 Feedback",
    "E": "Exercise Assessment Centre In-tray Role Play",
    "K": "Knowledge Skills Technical Programming Coding",
    "M": "Motivation Motivational Values",
    "P": "Personality Behaviour Behavior Trait Psychometric",
    "S": "Simulation Simulated Work-sample",
}

def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [w for w in text.split() if len(w) > 1]

def make_doc(item: dict) -> str:
    type_expansions = " ".join(
        TEST_TYPE_LABELS.get(t, t) for t in item.get("test_type", [])
    )
    langs = ", ".join(item.get("languages", []))
    duration = item.get("duration", "")
    remote = "remote online" if item.get("remote_testing") else ""
    adaptive = "adaptive IRT" if item.get("adaptive") else ""
    return (
        f"{item['name']} {item['name']} "          # name twice for boost
        f"{item.get('description', '')} "
        f"{type_expansions} "
        f"duration {duration} "
        f"languages {langs} "
        f"{remote} {adaptive}"
    ).strip()

class Retriever:
    def __init__(self, catalog: List[Dict]):
        self.catalog = catalog
        docs = [make_doc(item) for item in catalog]
        tokenized = [tokenize(d) for d in docs]
        self.bm25 = BM25Okapi(tokenized)
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=8000)
        self.tfidf_matrix = self.tfidf.fit_transform(docs)
        self.docs = docs

    def search(self, query: str, k: int = 10) -> List[Dict]:
        tokens = tokenize(query)

        # BM25 scores
        bm25_scores = np.array(self.bm25.get_scores(tokens))

        # TF-IDF cosine scores
        q_vec = self.tfidf.transform([query])
        tfidf_scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()

        # Normalize each to [0,1]
        def norm(arr):
            mn, mx = arr.min(), arr.max()
            return (arr - mn) / (mx - mn + 1e-9)

        combined = 0.5 * norm(bm25_scores) + 0.5 * norm(tfidf_scores)
        top_idx = np.argsort(combined)[::-1][:k]

        results = []
        for i in top_idx:
            item = dict(self.catalog[i])
            item["_score"] = float(combined[i])
            results.append(item)
        return results

def build_and_save():
    catalog = json.loads(CATALOG_PATH.read_text())
    r = Retriever(catalog)
    INDEX_PATH.write_bytes(pickle.dumps(r))
    print(f"✓ Retriever built over {len(catalog)} items → {INDEX_PATH}")
    return r

def load() -> Retriever:
    if INDEX_PATH.exists():
        return pickle.loads(INDEX_PATH.read_bytes())
    return build_and_save()

if __name__ == "__main__":
    build_and_save()
