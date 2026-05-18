# build-hybrid-gpt



### Wikipedia dataset
Download **20220301.en** shard of [aitetic/wikipedia](https://huggingface.co/datasets/aitetic/wikipedia) dataset:
```bash
hf download aitetic/wikipedia --repo-type dataset --include "20220301.en/*" --local-dir ./datasets/wikipedia
```

* Rows: 6_458_670

