import json
from datasets import load_dataset


OUTPUT_FILE = "dataset.jsonl"


def iter_reddit():
    ds = load_dataset("aitetic/reddit-17", split="train")

    for row in ds:
        text = row.get("content", "")
        if text:
            yield {
                "src": "reddit-17",
                "text": text.strip()
            }


def iter_eli5():
    ds = load_dataset("aitetic/eli5-lfqa", split="train")
    ds = ds.flatten()   # answers.text -> обычное поле

    for row in ds:
        question = (row.get("question") or "").strip()
        answers = row.get("answers.text", [])

        if not question:
            continue

        parts = [question + "?"]

        for ans in answers:
            ans = ans.strip()
            if ans:
                parts.append(ans)

        yield {
            "src": "eli5-lfqa",
            "text": "\n".join(parts)
        }


def main():
    count = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        for record in iter_reddit():
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

        for record in iter_eli5():
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"Done. Wrote {count:,} records to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
