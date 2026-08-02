# build-hybrid-llama

* run `wikipedia.py` to download and prepare the wikipedia dataset
* run `main_ddp.py` for a distributed data parallel (DDP) pre-training loop across multiple GPUs


Tune params in `main_ddp.py`:
```python
#total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
#B = 64 # micro batch size
total_batch_size = 540672 # 2**19, ~0.5M, in number of tokens
B = 16 # micro batch size
```

The trained model will be located in `train_products` directory.


## GPT-R (GPT-RoPE)
GPT-R is hybrid of nano-GPT model with RoPE technique.


#### Runpod.io set-up:

* Optionally:
```bash
pip install torch==2.8.0 --extra-index-url https://download.pytorch.org/whl/cu126
```

* Regular installation:
```bash
pip install datasets==3.6.0
pip install transformers==4.56.1
pip install huggingface-hub==0.34.3
pip install accelerate==1.10.1
```
