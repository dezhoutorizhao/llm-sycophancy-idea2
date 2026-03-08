import os
import re
import ssl
import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

BASE_DIR = r"e:\谄媚KVcache_idea\Sycopancy Interpretability代码\sycophancy-interpretability\papers_kv_cache_jailbreak"
INDEX_PATH = os.path.join(BASE_DIR, "index.json")

QUERIES = [
    ("PROMPTPEEK", "PROMPTPEEK KV Cache timing side-channel LLM"),
    ("CacheAttack", "CacheAttack semantic caching key collision LLM"),
    ("CacheTrap", "CacheTrap Data-and Gradient-free LLM Trojan KV cache"),
    ("StreamingLLM", "StreamingLLM"),
    ("H2O", "H2O Heavy-Hitter Oracle for Efficient Generative Inference"),
    ("ADC", "Adaptive Dense-to-Sparse Constrained Optimization jailbreak"),
    ("AttnGCG", "Attention-manipulated GCG jailbreak"),
    ("MetaBreak", "MetaBreak special token jailbreak"),
    ("ArtPrompt", "ArtPrompt ASCII art jailbreak"),
    ("SATA", "Simple Auxiliary Tasks jailbreak LLM"),
    ("DeepInception", "DeepInception jailbreak"),
    ("RobustKV", "RobustKV jailbreak defense"),
    ("Hidden State Filtering", "Hidden State Filtering jailbreak defense"),
    ("SafeProbing", "SafeProbing jailbreak"),
    ("SafeDecoding", "Defending Against Jailbreak Attacks via Safety-Aware Decoding"),
    ("Backtranslation", "Defending Language Models Against Jailbreak Attacks via Backtranslation"),
    ("Intention Analysis", "Intention Analysis jailbreak LLM self-correction"),
    ("SmoothLLM", "SmoothLLM Defending Large Language Models Against Jailbreaking Attacks"),
    ("Erase-and-Check", "Erase-and-Check jailbreak defense"),
    ("RedBench", "RedBench benchmark jailbreak"),
    ("JailbreakRadar", "JailbreakRadar"),
    ("The Instruction Hierarchy", "The Instruction Hierarchy"),
    ("FocalLoRA", "FocalLoRA"),
]


def sanitize(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", name).strip()[:120]


def arxiv_search(query: str):
    encoded = urllib.parse.quote(query)
    url = f"https://export.arxiv.org/api/query?search_query=all:{encoded}&start=0&max_results=1"
    with urllib.request.urlopen(url, timeout=25, context=ssl.create_default_context()) as resp:
        data = resp.read()
    root = ET.fromstring(data)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    entry = root.find("a:entry", ns)
    if entry is None:
        return None
    title = " ".join((entry.find("a:title", ns).text or "").split())
    aid = (entry.find("a:id", ns).text or "").strip().split("/")[-1]
    if not aid:
        return None
    return {
        "source": "arxiv",
        "id": aid,
        "title": title,
        "pdf_url": f"https://arxiv.org/pdf/{aid}.pdf",
        "abs_url": f"https://arxiv.org/abs/{aid}",
    }


def download_file(url: str, output_path: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=40, context=ssl.create_default_context()) as resp:
        content = resp.read()
    with open(output_path, "wb") as f:
        f.write(content)
    return len(content)


def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    records = []
    for alias, query in QUERIES:
        item = {"alias": alias, "query": query, "status": "failed"}
        try:
            hit = arxiv_search(query)
            if not hit:
                item["error"] = "no_arxiv_hit"
                records.append(item)
                continue
            filename = sanitize(f"{alias}__{hit['id']}.pdf")
            save_path = os.path.join(BASE_DIR, filename)
            size = download_file(hit["pdf_url"], save_path)
            item.update(
                {
                    "status": "downloaded",
                    "title": hit["title"],
                    "arxiv_id": hit["id"],
                    "pdf_url": hit["pdf_url"],
                    "abs_url": hit["abs_url"],
                    "file_path": save_path,
                    "bytes": size,
                }
            )
        except Exception as e:
            item["error"] = str(e)
        records.append(item)
        print(f"{alias}: {item['status']}")

    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    ok = sum(1 for r in records if r["status"] == "downloaded")
    print(f"downloaded={ok}/{len(records)}")
    print(INDEX_PATH)


if __name__ == "__main__":
    main()
