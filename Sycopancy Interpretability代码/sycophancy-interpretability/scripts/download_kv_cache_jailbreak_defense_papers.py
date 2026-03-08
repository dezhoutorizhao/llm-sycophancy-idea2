import json
import os
import re
import ssl
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime

PROJECT_ROOT = r"e:\谄媚KVcache_idea\Sycopancy Interpretability代码\sycophancy-interpretability"
OUT_DIR = os.path.join(PROJECT_ROOT, "papers", "kv_cache_jailbreak_defense")
INDEX_JSON = os.path.join(OUT_DIR, "index.json")

TOPICS = [
    ("PROMPTPEEK", "PROMPTPEEK kv cache side-channel llm"),
    ("CacheTrap", "CacheTrap kv cache llm trojan"),
    ("RobustKV", "RobustKV jailbreak defense kv cache"),
    ("kv-cache eviction defense", "jailbreak defense kv cache eviction"),
    ("kv-cache compression risk", "kv cache compression selective amnesia llm"),
    ("semantic cache collision", "semantic caching llm cache collision attack"),
]


def sanitize_filename(text: str) -> str:
    text = re.sub(r'[\\/:*?"<>|]+', "_", text)
    text = re.sub(r"\s+", "_", text).strip("_")
    return text[:140]


def arxiv_query(query: str, max_results: int = 5):
    encoded = urllib.parse.quote(query)
    url = (
        "https://export.arxiv.org/api/query?"
        f"search_query=all:{encoded}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    )
    with urllib.request.urlopen(url, timeout=30, context=ssl.create_default_context()) as resp:
        payload = resp.read()
    root = ET.fromstring(payload)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    entries = []
    for e in root.findall("a:entry", ns):
        title = " ".join((e.find("a:title", ns).text or "").split())
        aid = (e.find("a:id", ns).text or "").strip().split("/")[-1]
        published = (e.find("a:published", ns).text or "").strip()
        authors = [n.find("a:name", ns).text for n in e.findall("a:author", ns)]
        if aid:
            entries.append(
                {
                    "id": aid,
                    "title": title,
                    "published": published,
                    "authors": authors,
                    "abs_url": f"https://arxiv.org/abs/{aid}",
                    "pdf_url": f"https://arxiv.org/pdf/{aid}.pdf",
                }
            )
    return entries


def pick_entry(entries, topic_name: str):
    if not entries:
        return None
    key = topic_name.lower()
    for e in entries:
        if key in e["title"].lower():
            return e
    return entries[0]


def download_pdf(url: str, path: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60, context=ssl.create_default_context()) as resp:
        data = resp.read()
    with open(path, "wb") as f:
        f.write(data)
    return len(data)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for topic_name, query in TOPICS:
        row = {
            "topic": topic_name,
            "query": query,
            "status": "failed",
            "fetched_at": datetime.utcnow().isoformat() + "Z",
        }
        try:
            entries = arxiv_query(query, max_results=5)
            chosen = pick_entry(entries, topic_name)
            if not chosen:
                row["error"] = "no_result"
                rows.append(row)
                print(f"{topic_name}: failed(no_result)")
                continue
            filename = sanitize_filename(
                f"{topic_name}__{chosen['published'][:10]}__{chosen['id']}.pdf"
            )
            save_path = os.path.join(OUT_DIR, filename)
            file_size = download_pdf(chosen["pdf_url"], save_path)
            row.update(chosen)
            row["file_path"] = save_path
            row["bytes"] = file_size
            row["status"] = "downloaded"
            print(f"{topic_name}: downloaded")
        except Exception as e:
            row["error"] = str(e)
            print(f"{topic_name}: failed({e})")
        rows.append(row)
    with open(INDEX_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    ok = sum(1 for r in rows if r["status"] == "downloaded")
    print(f"downloaded={ok}/{len(rows)}")
    print(INDEX_JSON)


if __name__ == "__main__":
    main()
