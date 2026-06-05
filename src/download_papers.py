import requests, os, time

PAPERS = [
    ("1706.03762", "attention_is_all_you_need"),
    ("1810.04805", "bert"),
    ("2005.11401", "rag_original_paper"),
    ("2304.01852", "generative_agents"),
    ("2303.08774", "gpt4_technical_report"),
    ("2112.09332", "chain_of_thought"),
    ("2201.11903", "self_consistency"),
    ("2210.11610", "react_reasoning"),
    ("2302.13971", "llama"),
    ("2307.09288", "llama2"),
    ("2305.10601", "voyager"),
    ("2309.10668", "mistral"),
    ("2306.07929", "textbooks_are_all_you_need"),
    ("2210.03629", "retrieval_lm_survey"),
    ("2312.10997", "rag_survey"),
    ("2310.11511", "loftq"),
    ("2305.14314", "gorilla"),
    ("2308.09687", "longllmlingua"),
    ("2301.12652", "self_instruct"),
    ("2303.17580", "reflexion"),
]

OUT = os.path.join("data", "raw")
os.makedirs(OUT, exist_ok=True)

for arxiv_id, name in PAPERS:
    path = os.path.join(OUT, f"{name}.pdf")
    if os.path.exists(path):
        print(f"  Already have: {name}")
        continue
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    print(f"  Downloading: {name}...")
    r = requests.get(url, timeout=30)
    if r.status_code == 200:
        with open(path, "wb") as f:
            f.write(r.content)
        print(f"  Saved: {name}.pdf ({len(r.content)//1024} KB)")
    else:
        print(f"  Failed: {name} (status {r.status_code})")
    time.sleep(1)

print(f"\nDone. Papers in: {OUT}")