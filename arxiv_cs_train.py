"""
Short-context warmup - train on first 100 tokens of each dataset item using a small learning rate as warmup before full pre-training.
This is a simple way to "prime" the model's weights and can lead to faster convergence and better performance in the early stages of training.
"""

import json
import os
import torch, math, random, numpy as np
from dataclasses import dataclass
from auto_config import AutoConfigModel
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from transformers import set_seed, GPT2TokenizerFast

from datasets import load_dataset

import matplotlib.pyplot as plt


SAVE_DIR = "train_products"
FILE_NAME = "model_trained.pt"
MODEL_REPO = "aitetic/gpt-r-0.3b"

DATASET_FILE = "./datasets/ELI5.jsonl"
DATASET_SPLIT = "train"
TEXT_KEY = "answers"


MAX_LEN = 512
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 1


LEARNING_RATE = 8e-5
LENGTH_HIST_BINS = 120
LENGTH_HIST_FILE = os.path.join(SAVE_DIR, "corpus_eli5_token_length_hist.png")


@dataclass
class TrainerConfig:
    epochs: int = 1
    batch_size: int = 4
    learning_rate: float = 8e-5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    grad_accum_steps: int = 1



def custom_collate_fn(batch, max_seq_length, pad_token_id, eos_token_id, device, ignore_index=-100):
    """
    Custom collate function for variable-length text samples.

    Args:
        batch: list of tokenized samples
        eos_token_id: int, used for padding termination
        device: torch.device

    Returns:
        inputs_tensor: [batch_size, seq_len]
        targets_tensor: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len]
        dataset_token_count: int, number of tokenized dataset tokens before EOS/padding
    """

    dataset_token_count = sum(int(item.numel()) for item in batch)

    # Add EOS only when the sample is shorter than the warmup cap.
    batch_max_length = max(len(item) + (1 if len(item) < max_seq_length else 0) for item in batch)

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []
    attn_lst = []

    for item in batch:

        new_item = item.tolist()
        if len(item) < max_seq_length:
            new_item.append(eos_token_id)
        real_len = len(new_item)

        # Pad sequences to max_length
        padded = new_item + [pad_token_id] * (batch_max_length - real_len)

        # build attention mask from real_len (NOT from token values)
        attn = [1] * real_len + [0] * (batch_max_length - real_len)

        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        am = torch.tensor(attn[:-1], dtype=torch.long)

        # Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if max_seq_length is not None:
            inputs = inputs[:max_seq_length]
            targets = targets[:max_seq_length]
            am = am[:max_seq_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)
        attn_lst.append(am)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    attention_mask = torch.stack(attn_lst).to(device)
    return inputs_tensor, targets_tensor, attention_mask, dataset_token_count


class JsonlDataset(IterableDataset):

    def __init__(self, hf_dataset, tokenizer, max_seq_length=MAX_LEN, max_rows=None, text_key="text"):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_rows = max_rows
        self.text_key = text_key

        total_rows = len(self.hf_dataset)
        self.total_rows = min(total_rows, max_rows) if max_rows is not None else total_rows

        print(
            f"JsonlDataset::loaded rows.sz={self.total_rows}, max_rows={self.max_rows}, max_seq_length={self.max_seq_length}"
        )

    def __len__(self):
        return self.total_rows

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        dataset = self.hf_dataset

        if self.max_rows is not None:
            dataset = dataset.select(range(self.total_rows))

        if worker_info is not None:
            dataset = dataset.shard(num_shards=worker_info.num_workers, index=worker_info.id, contiguous=True)

        for row in dataset:
            text = row.get(self.text_key, "")
            if text is None:
                text = ""
            elif not isinstance(text, str):
                text = str(text)

            yield self.tokenizer(
                text,
                truncation=True,
                add_special_tokens=False,
                max_length=self.max_seq_length,
                padding=False,
                return_tensors="pt",
            )["input_ids"].squeeze(0)


class Trainer:

    def __init__(self, model, dataset, config, tokenizer):
        self.losses = []
        self.step_losses = []
        self.epoch_dataset_token_counts = []
        self.dataset_tokens_processed = 0

        self.model = model.to(config.device).float()
        self.config = config
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.loader = DataLoader(
            dataset,
            batch_size = config.batch_size,
            shuffle=False,
            collate_fn=lambda batch: custom_collate_fn(
                batch,
                #max_seq_length = model.config.block_size,
                max_seq_length = MAX_LEN,
                pad_token_id = self.tokenizer.eos_token_id,
                eos_token_id = self.tokenizer.eos_token_id,
                device = config.device,
                ),
            )


    def train(self):

        torch.set_float32_matmul_precision("high")

        # 1) Gradient accumulation should be an explicit hyperparameter
        grad_accum_steps = int(getattr(self.config, "grad_accum_steps", 1))
        grad_accum_steps = max(1, grad_accum_steps)
        # if epoch has fewer batches than accum steps — clamp
        grad_accum_steps = min(grad_accum_steps, max(1, len(self.loader)))

        self.losses = []          # token-weighted epoch losses (good for PPL)
        self.step_losses = []     # avg per-window accumulation raw loss (for plotting)
        self.epoch_dataset_token_counts = []
        self.dataset_tokens_processed = 0

        self.model.train()
        for epoch in range(self.config.epochs):
            pbar = tqdm(self.loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}")

            total_loss_sum = 0.0   # sum of (mean_loss * num_valid_tokens)
            total_loss_tokens = 0  # number of non-ignored tokens
            total_dataset_tokens = 0
            first_loss = None

            self.optimizer.zero_grad(set_to_none=True)

            accum_raw_sum = 0.0

            for step, batch in enumerate(pbar):
                # NOTE: your collate already .to(device), so these .to() are redundant but harmless
                #x = x.to(self.config.device, non_blocking=True)
                #y = y.to(self.config.device, non_blocking=True)

                input_ids, labels, attention_mask, dataset_token_count = batch

                total_dataset_tokens += int(dataset_token_count)

                # Forward pass
                raw_loss = self.model(input_ids, labels).loss

                # ---- logging helpers ----
                raw = float(raw_loss.detach().cpu().item())
                accum_raw_sum += raw

                # token-weighted stats for correct epoch avg loss / PPL
                with torch.no_grad():
                    loss_token_count = int((labels != -100).sum().item())
                total_loss_sum += raw * loss_token_count
                total_loss_tokens += loss_token_count

                # ---- backward (accumulation) ----
                loss = raw_loss / grad_accum_steps
                loss.backward()

                # Progress bar smoothing
                if first_loss is None:
                    first_loss = raw
                    pbar.set_postfix(loss=f"{first_loss:.4f}", accum_steps=str(grad_accum_steps))


                # Optimizer step
                if ((step + 1) % grad_accum_steps == 0) or ((step + 1) == len(self.loader)):
                    if getattr(self.config, "max_grad_norm", None) is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.max_grad_norm))
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    # calculate the average raw loss for current accumulation window
                    step_avg_loss = accum_raw_sum / grad_accum_steps
                    accum_raw_sum = 0.0
                    self.step_losses.append(step_avg_loss)

                    pbar.set_postfix(loss=f"{step_avg_loss:.4f}", accum_steps=str(grad_accum_steps))


            # ---- epoch metrics (token-weighted, correct for variable lengths) ----
            if total_loss_tokens == 0:
                epoch_avg_loss = float("nan")
                ppl = float("nan")
            else:
                epoch_avg_loss = total_loss_sum / total_loss_tokens
                # # Calculate Perplexity, avoid overflow for huge losses
                ppl = math.exp(epoch_avg_loss) if epoch_avg_loss < 50 else float("inf")

            self.losses.append(epoch_avg_loss)
            self.epoch_dataset_token_counts.append(total_dataset_tokens)
            self.dataset_tokens_processed += total_dataset_tokens
            print(f"Epoch {epoch+1}: epoch_avg_loss={epoch_avg_loss:.4f}, PPL={ppl:.4f}, dataset_tokens={total_dataset_tokens:_}")

        print(
            "✅ Training completed,",
            f"steps: {len(self.step_losses)}, final_avg_loss: {self.losses[-1]:.4f}, dataset_tokens_processed={self.dataset_tokens_processed:_}"
        )

        return self.losses, self.step_losses


def plot_losses(losses1: list, label1: str, x_label: str):

    plt.plot(range(len(losses1)), losses1, label=label1, color="blue")

    plt.xlabel(x_label)
    plt.ylabel("Loss")
    plt.title(f"Training")
    plt.legend()

    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def analyze_token_length_distribution(
    hf_dataset,
    tokenizer,
    text_key: str = TEXT_KEY,
    max_rows: int | None = None,
    bins: int = LENGTH_HIST_BINS,
    output_path: str = LENGTH_HIST_FILE,
):
    """
    Compute token length for each record, print summary stats, and save histogram.
    """
    if max_rows is not None:
        rows = hf_dataset.select(range(min(len(hf_dataset), max_rows)))
    else:
        rows = hf_dataset

    lengths = []
    iterator = tqdm(rows, desc="Token length scan")

    for row in iterator:
        texts = row.get(text_key, [])
        text = " ".join(texts) if isinstance(texts, list) else texts
        text = text.strip() if isinstance(text, str) else str(text).strip()
        if text is None:
            text = ""

        input_ids = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]
        lengths.append(len(input_ids))

    if not lengths:
        print("No rows found for token length analysis")
        return []

    lengths_np = np.asarray(lengths, dtype=np.int32)
    p50, p90, p95, p99 = np.percentile(lengths_np, [50, 90, 95, 99])
    over_max_len = float((lengths_np > MAX_LEN).mean()) * 100.0

    print("Token length distribution summary:")
    print(f"  rows: {len(lengths_np):_}")
    print(f"  min: {int(lengths_np.min())}")
    print(f"  mean: {float(lengths_np.mean()):.2f}")
    print(f"  median (p50): {p50:.2f}")
    print(f"  p90: {p90:.2f}")
    print(f"  p95: {p95:.2f}")
    print(f"  p99: {p99:.2f}")
    print(f"  max: {int(lengths_np.max())}")
    print(f"  rows > MAX_LEN({MAX_LEN}): {over_max_len:.2f}%")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.hist(lengths_np, bins=bins, color="steelblue", edgecolor="black", alpha=0.85)
    plt.axvline(MAX_LEN, color="red", linestyle="--", linewidth=1.5, label=f"MAX_LEN={MAX_LEN}")
    plt.xlabel("Tokens per record")
    plt.ylabel("Count")
    plt.title("Corpus ELI5 token length distribution")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()

    print(f"Saved histogram to: {output_path}")
    return lengths


def run_warmup_stage(
    model,
    tokenizer,
    train_config,
    hf_dataset,
    max_rows=None,
):
    dataset = JsonlDataset(hf_dataset, tokenizer, max_seq_length=MAX_LEN, max_rows=max_rows)
    if len(dataset) == 0:
        return model, [], []

    trainer = Trainer(model, dataset, train_config, tokenizer)
    epoch_losses, step_losses = trainer.train()

    return trainer.model, epoch_losses, step_losses


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    train_config = TrainerConfig(learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, grad_accum_steps=GRAD_ACCUM_STEPS, device=device)

    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_REPO, local_files_only=False)

    model = AutoConfigModel.from_pretrained(
        MODEL_REPO,
        map_location=device,
        local_files_only=True,
    )
    if model is None:
        print(f"local cache miss for {MODEL_REPO}, downloading from Hugging Face Hub...")
        model = AutoConfigModel.from_pretrained(
            MODEL_REPO,
            map_location=device,
            local_files_only=False,
        )
    if model is None:
        raise SystemExit(f"Checkpoint '{MODEL_REPO}' was not found on Hugging Face Hub or in the local cache.")

    dataset_path = DATASET_FILE
    if not os.path.isfile(dataset_path):
        raise SystemExit(f"Dataset file was not found: {dataset_path}")

    print(f"Loading local dataset from: {dataset_path}")
    fw = load_dataset("json", data_files=dataset_path, split=DATASET_SPLIT)

    print(f"dataset: {dataset_path}, split: {DATASET_SPLIT}, text_key: {TEXT_KEY}")

    # print("Building token length distribution before training...")
    analyze_token_length_distribution(
        fw,
        tokenizer,
        text_key=TEXT_KEY,
        max_rows=None,
        bins=LENGTH_HIST_BINS,
        output_path=LENGTH_HIST_FILE,
    )

