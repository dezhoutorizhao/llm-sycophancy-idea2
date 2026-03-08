import json
import os
import re
import ssl
import urllib.request
from datetime import datetime

PROJECT_ROOT = r"e:\谄媚KVcache_idea\Sycopancy Interpretability代码\sycophancy-interpretability"
OUT_DIR = os.path.join(PROJECT_ROOT, "papers", "kv_cache_jailbreak_defense")
INDEX_PATH = os.path.join(OUT_DIR, "index_curated.json")

PAPERS = [
    {
        "id": "promptpeek_ndss2025",
        "title": "I Know What You Asked: Prompt Leakage via KV-Cache Sharing in Multi-Tenant LLM Serving",
        "year": 2025,
        "venue": "NDSS",
        "category": "KV-Cache Attack",
        "pdf_url": "https://www.ndss-symposium.org/wp-content/uploads/2025-1772-paper.pdf",
        "code_url": "",
    },
    {
        "id": "cachetrap_arxiv2511",
        "title": "CacheTrap: Injecting Trojans in LLMs without Leaving any Traces in Inputs or Weights",
        "year": 2025,
        "venue": "arXiv",
        "category": "KV-Cache Attack",
        "pdf_url": "https://arxiv.org/pdf/2511.22681.pdf",
        "code_url": "",
    },
    {
        "id": "robustkv_iclr2025",
        "title": "RobustKV: Defending Large Language Models against Jailbreak Attacks via KV Eviction",
        "year": 2025,
        "venue": "ICLR",
        "category": "KV-Cache Defense",
        "pdf_url": "https://proceedings.iclr.cc/paper_files/paper/2025/file/38bbae17d60940f3ee14dfd1035d7542-Paper-Conference.pdf",
        "code_url": "https://github.com/TanqiuJiang/RobustKV",
    },
    {
        "id": "cacheattack_arxiv2601",
        "title": "From Similarity to Vulnerability: Key Collision Attack on LLM Semantic Caching",
        "year": 2026,
        "venue": "arXiv",
        "category": "Semantic Cache Attack",
        "pdf_url": "https://arxiv.org/pdf/2601.23088.pdf",
        "code_url": "",
    },
    {
        "id": "earlybird_arxiv2409",
        "title": "The Early Bird Catches the Leak: Unveiling Timing Side Channels in LLM Serving Systems",
        "year": 2024,
        "venue": "arXiv",
        "category": "KV/Prompt Cache Side-channel",
        "pdf_url": "https://arxiv.org/pdf/2409.20002.pdf",
        "code_url": "",
    },
]


def slugify(text: str) -> str:
    text = re.sub(r"[\\/:*?\"<>|]+", "_", text)
    text = re.sub(r"\s+", "_", text).strip("_")
    return text[:120]


def download(url: str, out_path: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=90, context=ssl.create_default_context()) as resp:
        data = resp.read()
    with open(out_path, "wb") as f:
        f.write(data)
    return len(data)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    index_rows = []
    for p in PAPERS:
        file_name = f"{p['year']}_{slugify(p['venue'])}_{slugify(p['id'])}.pdf"
        abs_path = os.path.join(OUT_DIR, file_name)
        row = {
            **p,
            "saved_file": abs_path,
            "status": "failed",
            "fetched_at": datetime.utcnow().isoformat() + "Z",
        }
        try:
            size = download(p["pdf_url"], abs_path)
            row["status"] = "downloaded"
            row["bytes"] = size
            print(f"downloaded: {p['id']}")
        except Exception as e:
            row["error"] = str(e)
            print(f"failed: {p['id']} -> {e}")
        index_rows.append(row)
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index_rows, f, ensure_ascii=False, indent=2)
    ok = sum(1 for r in index_rows if r["status"] == "downloaded")
    print(f"downloaded={ok}/{len(index_rows)}")
    print(INDEX_PATH)


if __name__ == "__main__":
    main()
