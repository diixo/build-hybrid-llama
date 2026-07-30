import json
import os
import time

from datasets import DownloadConfig, load_dataset


OUTPUT_FILE = "dataset.jsonl"


def load_hf_dataset(dataset_name, split="train"):
    cache_dir = os.path.join(os.getcwd(), ".hf_cache", dataset_name.replace("/", "__"))
    os.makedirs(cache_dir, exist_ok=True)

    for attempt in range(3):
        try:
            return load_dataset(
                dataset_name,
                split=split,
                cache_dir=cache_dir,
                download_config=DownloadConfig(
                    max_retries=5,
                    resume_download=True,
                    disable_tqdm=False,
                ),
                download_mode="reuse_cache_if_exists",
            )
        except Exception as exc:
            if attempt == 2:
                raise
            print(f"[dataset_assembler] failed to load {dataset_name} on attempt {attempt + 1}: {exc}")
            time.sleep(10)


def iter_eli5(ds):

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
            "src": "qa",
            "text": "\n".join(parts)
        }


def iter_eli5_ctxs(ds):

    for row in ds:
        ctxs = row.get("ctxs", [])
        parts = []

        for ctx in ctxs:
            ctx = ctx.strip()
            if ctx:
                parts.append(ctx)

        yield {
            "src": "ctxs",
            "text": "\n".join(parts)
        }


def main():

    ds = load_hf_dataset("aitetic/eli5-lfqa", split="train")

    ds = ds.flatten()

    count = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        for record in iter_eli5_ctxs(ds):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

            if count % 1000 == 0:
                print(f"...{count:,} records from: eli5-lfqa")

        f.flush()

        for record in iter_eli5(ds):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

            if count % 1000 == 0:
                print(f"...{count:,} records from: eli5-lfqa")

        f.flush()

    print(f"Done. Wrote {count:,} records to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
