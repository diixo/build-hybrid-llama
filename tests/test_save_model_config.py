import json
import tempfile
from pathlib import Path

from modeling_gptr import GPTRForCausalLM, GPTConfig


def test_save_model_config_writes_config_to_output_dir_and_checkpoint_dir():
    model = GPTRForCausalLM(GPTConfig(vocab_size=32, n_layer=1, n_head=1, n_embd=8, block_size=16))

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "output"
        checkpoint_dir = output_dir / "checkpoint-10"

        model.save_model_config(output_dir)
        model.save_model_config(checkpoint_dir)

        assert (output_dir / "config.json").is_file()
        assert (checkpoint_dir / "config.json").is_file()

        config_data = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
        assert config_data["vocab_size"] == 32
