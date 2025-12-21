

import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['WANDB_API_KEY'] = 'your key' 
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from tqdm import tqdm
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 400
log_interval = 10
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = False # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'

# wandb logging
wandb_log = True # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())

# data
dataset = 'openwebtext' 
gradient_accumulation_steps = 60 # used to simulate larger batch sizes
batch_size = 8 # if gradient_accumulation_steps > 1, this is the micro-batch size
max_position_embeddings = 1024


# model
num_hidden_layers = 24
num_attention_heads = 16
num_key_value_heads=4
head_dim=64
hidden_size = 1024
intermediate_size=2560

dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?

# adamw optimizer
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
norm_eps=1e-6
rope_theta=10000


# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 800 # how many steps to warm up for
lr_decay_iters = 15000 # should be ~= max_iters per Chinchilla
learning_rate = 3.2e-4 # max learning rate
max_iters = 15000 # total number of training iterations
min_lr = 3.2e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster

master_process = True


# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------
print("参数检查----------------------------------------")

print(f"batch_size: {batch_size}")
print(f"gradient_accumulation_steps: {gradient_accumulation_steps}")
print(f"tokens per iteration: {gradient_accumulation_steps * batch_size * max_position_embeddings}")
print(f"eval_interval: {eval_interval}")
print(f"max_iters: {max_iters}")
print(f"learning_rate: {learning_rate}")
print(f"dtype: {dtype}")




tokens_per_iter = gradient_accumulation_steps *  batch_size * max_position_embeddings
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(45)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
# data_dir='../shakespeare_char' 
data_dir = os.path.join('data', dataset)

# def print_gpu_memory():
#     """
#     打印当前 GPU 显存占用详情
#     tag: 用于标记打印位置（例如 'Before Backward', 'After Optimizer'）
#     """
#     if not torch.cuda.is_available():
#         print("CUDA is not available.")
#         return

#     # 1 GB = 1024*1024*1024 Bytes
#     GB = 1024**3 
    
#     # Allocated: 实际被 Tensor 占用的显存（你的模型权重、梯度、激活值）
#     allocated = torch.cuda.memory_allocated() / GB
    
#     # Reserved: PyTorch 向系统申请的总显存（包含 allocated + 碎片/未使用的缓存）
#     reserved = torch.cuda.memory_reserved() / GB
    
#     #从程序开始（或上次重置）到现在的最大占用峰值
#     peak = torch.cuda.max_memory_allocated() / GB
    
#     print(f"Actual Usage (Allocated): {allocated:.2f} GB")
#     print(f" Total Cache  (Reserved):  {reserved:.2f} GB")
#     print(f" Peak Usage   (Max):       {peak:.2f} GB")



def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - max_position_embeddings, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+max_position_embeddings]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+max_position_embeddings]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(num_hidden_layers=config['num_hidden_layers'], 
                  num_attention_heads=config['num_attention_heads'], 
                  num_key_value_heads=config['num_key_value_heads'], 
                  head_dim=config['head_dim'], 
                  hidden_size=config['hidden_size'], 
                  max_position_embeddings=config['max_position_embeddings'],
                  bias=config['bias'], 
                  vocab_size=None, 
                  dropout=config['dropout'],
                  intermediate_size=config['intermediate_size'],
                  norm_eps=config['norm_eps'],
                  rope_theta=config['rope_theta']
                ) # start with model_args from command line

if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line

    for k in ['num_hidden_layers', 'num_attention_heads', 'num_key_value_heads', 'head_dim', 'hidden_size', 'max_position_embeddings','bias', 'vocab_size', 'dropout', 'intermediate_size', 'norm_eps', 'rope_theta']:
        model_args[k] = checkpoint_model_args[k]

    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

if max_position_embeddings < model.config.max_position_embeddings:
    model.crop_block_size(max_position_embeddings)
    model_args['max_position_embeddings'] = max_position_embeddings # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
unoptimized_model = model
if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model) # requires PyTorch 2.0


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx: #with torch.amp.autocast(device_type=device_type, dtype=ptdtype)
                logits, loss = model(X, Y)
                del logits
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()

    torch.cuda.empty_cache()
    return out  #out是一个字典，包含训练集和验证集的平均损失

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    print("wandb logging enabled")
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
running_mfu = -1.0


#加入进度条...
pbar = tqdm(total=max_iters, initial=iter_num, desc="Training", dynamic_ncols=True)
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    # print("optimizer.param_groups:", optimizer.param_groups)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        #计算loss前先清理梯度
        optimizer.zero_grad(set_to_none=True)

        losses = estimate_loss()
        log_message = f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        tqdm.write(log_message)
        if wandb_log:
            try:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu*100, # convert to percentage
                })
            except Exception as e:
                print(f"wandb log error: {e}")
                wandb_log = False
                pass
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': unoptimized_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                tqdm.write(f"saving checkpoint")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = unoptimized_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        # print_gpu_memory()
        pbar.set_postfix({
            "loss": f"{lossf:.4f}", 
            "time": f"{dt*1000:.1f}ms", 
            "mfu": f"{running_mfu*100:.1f}%",
            "lr": f"{lr:.2e}"
        })
    iter_num += 1
    local_iter_num += 1
    pbar.update(1)

    # termination conditions
    if iter_num > max_iters:
        break
pbar.close()