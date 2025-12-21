import time
wandb_log = True
wandb_project = 'easygpt'
wandb_run_name = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
msx_position_embeddings = 1024
gradient_accumulation_steps = 40

# this makes total number of tokens be 300B
max_iters = 15000
lr_decay_iters = 15000

# eval stuff
eval_interval = 500
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1



