"""
Fine-tune the current GPT-R checkpoint on the OpenOrca instruction dataset.

The script uses a streaming Hugging Face dataset and supervised fine-tuning loss:
prompt tokens are masked out, and the model is trained only on assistant tokens.
"""

import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import math
import torch
from dataclasses import dataclass
from itertools import islice
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import GPT2TokenizerFast

from auto_config import AutoConfigModel


SAVE_DIR = "train_products"
FILE_NAME = "model-open-orca.pt"

DATASET_REPO = "Open-Orca/OpenOrca"
MODEL_REPO = "aitetic/gpt-r-0.3b-warmup"
TOKENIZER_PATH = "data/gpt-noomo-32k"

IM_START_TOKEN = "<|im_start|>"
IM_END_TOKEN = "<|im_end|>"


@dataclass
class TrainerConfig:
	epochs: int = 1
	steps_per_epoch: int = 1000
	batch_size: int = 4
	learning_rate: float = 5e-5
	device: str = "cuda" if torch.cuda.is_available() else "cpu"
	grad_accum_steps: int = 1
	max_seq_length: int = 1024
	num_workers: int = 0
	max_grad_norm: float | None = 1.0


def _normalize_text(value) -> str:
	if value is None:
		return ""
	if not isinstance(value, str):
		value = str(value)
	return value.strip()


def format_open_orca_record(row) -> tuple[str, str] | None:
	system_prompt = _normalize_text(row.get("system_prompt"))
	question = _normalize_text(row.get("question"))
	response = _normalize_text(row.get("response"))

	if not question or not response:
		return None

	prompt_parts = []
	if system_prompt:
		prompt_parts.append(f"{IM_START_TOKEN}system\n{system_prompt}{IM_END_TOKEN}")
	prompt_parts.append(f"{IM_START_TOKEN}user\n{question}{IM_END_TOKEN}")
	prompt_parts.append(f"{IM_START_TOKEN}assistant\n")

	prompt_text = "\n".join(prompt_parts)
	full_text = prompt_text + response + IM_END_TOKEN
	return prompt_text, full_text


class OpenOrcaDataset(IterableDataset):

	def __init__(self, tokenizer, max_seq_length, max_rows=None, split="train", streaming=True):
		self.tokenizer = tokenizer
		self.max_seq_length = max_seq_length
		self.max_rows = max_rows
		self.split = split
		self.streaming = streaming

		print(
			f"OpenOrcaDataset::loaded split={self.split}, streaming={self.streaming}, "
			f"max_rows={self.max_rows}, max_seq_length={self.max_seq_length}"
		)

	def __iter__(self):
		worker_info = torch.utils.data.get_worker_info()
		dataset = load_dataset(DATASET_REPO, split=self.split, streaming=self.streaming)

		if worker_info is not None:
			dataset = dataset.shard(num_shards=worker_info.num_workers, index=worker_info.id)

		valid_rows = 0

		for row in iter(dataset):
			formatted = format_open_orca_record(row)
			if formatted is None:
				continue

			prompt_text, full_text = formatted

			prompt_ids = self.tokenizer(
				prompt_text,
				truncation=True,
				add_special_tokens=False,
				max_length=self.max_seq_length,
				padding=False,
				return_tensors="pt",
			)["input_ids"].squeeze(0)

			full_ids = self.tokenizer(
				full_text,
				truncation=True,
				add_special_tokens=False,
				max_length=self.max_seq_length,
				padding=False,
				return_tensors="pt",
			)["input_ids"].squeeze(0)

			if full_ids.numel() < 2:
				continue

			prompt_len = min(int(prompt_ids.numel()), int(full_ids.numel()))
			if prompt_len >= int(full_ids.numel()):
				continue

			yield {
				"input_ids": full_ids,
				"prompt_len": prompt_len,
			}

			valid_rows += 1
			if self.max_rows is not None and valid_rows >= self.max_rows:
				break


def custom_collate_fn(batch, max_seq_length, pad_token_id, eos_token_id, device, ignore_index=-100):
	normalized_batch = []
	for item in batch:
		token_ids = item["input_ids"]
		if not isinstance(token_ids, torch.Tensor):
			token_ids = torch.tensor(token_ids, dtype=torch.long)
		normalized_batch.append({
			"input_ids": token_ids,
			"prompt_len": int(item["prompt_len"]),
		})

	dataset_token_count = sum(int(item["input_ids"].numel()) for item in normalized_batch)

	batch_max_length = max(
		len(item["input_ids"]) + (1 if len(item["input_ids"]) < max_seq_length else 0)
		for item in normalized_batch
	)

	inputs_lst, targets_lst, attn_lst = [], [], []

	for item in normalized_batch:
		token_ids = item["input_ids"]
		prompt_len = int(item["prompt_len"])

		sequence = token_ids.tolist()
		if len(token_ids) < max_seq_length:
			sequence.append(eos_token_id)
		real_len = len(sequence)

		padded = sequence + [pad_token_id] * (batch_max_length - real_len)
		attn = [1] * real_len + [0] * (batch_max_length - real_len)

		inputs = torch.tensor(padded[:-1], dtype=torch.long)
		targets = torch.tensor(padded[1:], dtype=torch.long)
		attention_mask = torch.tensor(attn[:-1], dtype=torch.long)
		target_valid = torch.tensor(attn[1:], dtype=torch.bool)

		targets[~target_valid] = ignore_index

		prompt_target_count = max(0, min(prompt_len - 1, targets.numel()))
		if prompt_target_count > 0:
			targets[:prompt_target_count] = ignore_index

		inputs_lst.append(inputs)
		targets_lst.append(targets)
		attn_lst.append(attention_mask)

	inputs_tensor = torch.stack(inputs_lst).to(device)
	targets_tensor = torch.stack(targets_lst).to(device)
	attention_tensor = torch.stack(attn_lst).to(device)
	return inputs_tensor, targets_tensor, attention_tensor, dataset_token_count


class Trainer:

	def __init__(self, model, dataset, config, tokenizer):
		self.model = model.to(config.device).float()
		self.config = config
		self.tokenizer = tokenizer
		self.device_type = "cuda" if str(config.device).startswith("cuda") else "cpu"
		self.optimizer = self.model.configure_optimizers(
			weight_decay=0.1,
			learning_rate=config.learning_rate,
			device_type=self.device_type,
			master_process=True,
		)
		self.loader = DataLoader(
			dataset,
			batch_size=config.batch_size,
			shuffle=False,
			num_workers=config.num_workers,
			collate_fn=lambda batch: custom_collate_fn(
				batch,
				max_seq_length=config.max_seq_length,
				pad_token_id=self.tokenizer.pad_token_id,
				eos_token_id=self.tokenizer.eos_token_id,
				device=config.device,
			),
		)
		self.loader_iter = None
		self.losses = []
		self.step_losses = []
		self.dataset_tokens_processed = 0

	def _next_batch(self):
		if self.loader_iter is None:
			self.loader_iter = iter(self.loader)
		try:
			return next(self.loader_iter)
		except StopIteration:
			self.loader_iter = iter(self.loader)
			try:
				return next(self.loader_iter)
			except StopIteration as exc:
				raise RuntimeError(
					"OpenOrcaDataset produced no valid batches. Increase max_rows or max_seq_length."
				) from exc

	def train(self):
		torch.set_float32_matmul_precision("high")

		self.losses = []
		self.step_losses = []
		self.dataset_tokens_processed = 0

		self.model.train()
		for epoch in range(self.config.epochs):
			pbar = tqdm(range(self.config.steps_per_epoch), desc=f"Epoch {epoch + 1}/{self.config.epochs}")

			total_loss_sum = 0.0
			total_loss_tokens = 0
			total_dataset_tokens = 0

			self.optimizer.zero_grad(set_to_none=True)

			for _ in pbar:
				accum_raw_sum = 0.0

				for _ in range(self.config.grad_accum_steps):
					input_ids, labels, attention_mask, dataset_token_count = self._next_batch()

					with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16, enabled=self.device_type == "cuda"):
						raw_loss = self.model(input_ids, labels, attention_mask=attention_mask).loss

					raw = float(raw_loss.detach().cpu().item())
					valid_tokens = int((labels != -100).sum().item())

					total_loss_sum += raw * valid_tokens
					total_loss_tokens += valid_tokens
					total_dataset_tokens += int(dataset_token_count)

					accum_raw_sum += raw
					(raw_loss / self.config.grad_accum_steps).backward()

				if self.config.max_grad_norm is not None:
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.max_grad_norm))

				self.optimizer.step()
				self.optimizer.zero_grad(set_to_none=True)

				step_avg_loss = accum_raw_sum / self.config.grad_accum_steps
				self.step_losses.append(step_avg_loss)
				pbar.set_postfix(loss=f"{step_avg_loss:.4f}")

			epoch_avg_loss = total_loss_sum / total_loss_tokens if total_loss_tokens > 0 else float("nan")
			ppl = math.exp(epoch_avg_loss) if epoch_avg_loss < 50 else float("inf")

			self.losses.append(epoch_avg_loss)
			self.dataset_tokens_processed += total_dataset_tokens

			print(
				f"Epoch {epoch + 1}: epoch_avg_loss={epoch_avg_loss:.4f}, "
				f"PPL={ppl:.4f}, dataset_tokens={total_dataset_tokens:_}"
			)

		print(
			"Training completed,",
			f"steps: {len(self.step_losses)}, final_avg_loss: {self.losses[-1]:.4f}, "
			f"dataset_tokens_processed={self.dataset_tokens_processed:_}"
		)

		return self.losses, self.step_losses


def run_open_orca_stage(model, tokenizer, train_config, max_rows=None):
	dataset = OpenOrcaDataset(
		tokenizer,
		max_seq_length=train_config.max_seq_length,
		max_rows=max_rows,
	)
	trainer = Trainer(model, dataset, train_config, tokenizer)
	epoch_losses, step_losses = trainer.train()
	return trainer.model, epoch_losses, step_losses


if __name__ == "__main__":
	tokenizer = GPT2TokenizerFast.from_pretrained(TOKENIZER_PATH, local_files_only=True)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	train_config = TrainerConfig()

	model = AutoConfigModel.from_pretrained(MODEL_REPO, map_location=train_config.device)
	if model is None:
		raise SystemExit(f"Checkpoint '{MODEL_REPO}' was not found on Hugging Face Hub or in the local cache.")

	print(f"using device: {train_config.device}")
	print(f"model.sz={model.get_num_params()}")

	smoke_rows = None

	model, epoch_losses, step_losses = run_open_orca_stage(
		model,
		tokenizer,
		train_config,
		max_rows=smoke_rows,
	)

	extra_info = {
		"tokenizer_path": TOKENIZER_PATH,
		"dataset_repo": DATASET_REPO,
		"epoch_losses": epoch_losses,
		"step_losses": step_losses,
	}
	model.save_model(SAVE_DIR, file_name=FILE_NAME, train_config=train_config, **extra_info)
