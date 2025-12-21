"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
#一个decoder包括embedding，位置编码、多个block，归一化层，线性层
#一个block包括3个部分 归一化层，多头注意力层，MLP层，整个GPT模型由多个block堆叠而成


class RMSNorm(nn.Module):
    """RMSNorm"""
    def __init__(self,dim,norm_eps:float=1e-6):
        super().__init__()
        self.norm_eps=norm_eps
        self.weight=nn.Parameter(torch.ones(dim))  #可训练的参数
    def _norm(self,x):
        return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.norm_eps)
    def forward(self,x):
        output = self._norm(x.float()).type_as(x)
        return output*self.weight
#不使用layernorm
# class LayerNorm(nn.Module):
#     """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

#     def __init__(self, ndim, bias):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(ndim))
#         self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

#     def forward(self, input):
#         return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        # dim 是每个头的大小 head_dim
        # 计算逆频率 inv_freq
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        
        # 预计算 cos 和 sin 缓存
        t = torch.arange(self.max_seq_len_cached).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq) # [T, dim/2]
        # Llama 风格的拼接方式，把 freqs 复制两份
        emb = torch.cat((freqs, freqs), dim=-1) # [T, dim]
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, seq_len=None):
        # 这里的 x 通常只是为了拿 device 和 dtype
        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...],
        )
def rotate_half(x):
    """把向量切半并重组，用于旋转逻辑"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k 的形状通常是 [B, n_head, T, head_dim]
    # cos, sin 的形状通常是 [1, 1, T, head_dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
#多头注意力层GQA
def repeat_kv(hidden_states,n_rep):
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        #一个查询有多少个k_v值
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        #GQA
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.bias
        )
        #旋转位置编码层,用于得到sin和cos
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=config.max_position_embeddings,base=config.rope_theta)
        # key, query, value projections for all heads, but in a batch
        # self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q=self.q_proj(x)
        k=self.k_proj(x)
        v=self.v_proj(x)
        q_states = q.view(B, T, -1, self.head_dim).transpose(1, 2) # (B, nh, T, hd)
        k_states = k.view(B, T, -1, self.head_dim).transpose(1, 2) # (B, nkv, T, hd)
        v_states = v.view(B, T, -1, self.head_dim).transpose(1, 2) # (B, nkv, T, hd)
        
        cos,sin=self.rotary_emb(T)
        cos = cos.to(q.dtype)
        sin = sin.to(q.dtype)
        q_states, k_states=apply_rotary_pos_emb(q_states, k_states, cos, sin)


        k_states=repeat_kv(k_states,self.num_key_value_groups)
        v_states=repeat_kv(v_states,self.num_key_value_groups)
#所以在上面这个地方kqv都是num_heads个
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            #这里使用了flash attention
            y = torch.nn.functional.scaled_dot_product_attention(q_states, k_states, v_states, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q_states @ k_states.transpose(-2, -1)) * (1.0 / math.sqrt(k_states.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.o_proj(y))
        return y

#MLP层
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size
        self.gate_proj  = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.bias)
        self.up_proj  = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.bias)
        self.down_proj  = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        x = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        x=self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size,config.norm_eps)
        self.self_attn = CausalSelfAttention(config)
        self.post_attetion_layernorm = RMSNorm(config.hidden_size,config.norm_eps)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attetion_layernorm(x))
        return x

@dataclass
class GPTConfig:
    max_position_embeddings: int = 1024

    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    hidden_size: int = 1024

    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads:int=4
    head_dim:int=64

    dropout: float = 0.0
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    intermediate_size:int=2560

    norm_eps:float=1e-6
    rope_theta:int=10000
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        # assert config.vocab_size is not None
        # assert config.block_size is not None
        self.config = config
        self.embed_tokens=nn.Embedding(config.vocab_size,config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers=nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)])

        self.norm=RMSNorm(config.hidden_size,config.norm_eps)
        
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.embed_tokens.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying  #其实本质十分容易理解

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('o_proj.weight') or pn.endswith('down_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.num_hidden_layers))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # device = idx.device
        b, t = idx.size()
        assert t <= self.config.max_position_embeddings, f"Cannot forward sequence of length {t}, block size is only {self.config.max_position_embeddings}"

        # forward the GPT model itself
        #  #token embeddings of shape (b, t, n_embd)
        x = self.dropout(self.embed_tokens(idx))
        for block in self.layers:
            x = block(x)
        x = self.norm(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits =  self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
    
#将模型进行缩小

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.max_position_embeddings
        self.config.max_position_embeddings = block_size
        # self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.layers:
            #这里是找掩码矩阵
            if hasattr(block.self_attn, 'bias'):
                block.self_attn.bias = block.self_attn.bias[:,:,:block_size,:block_size]

    # @classmethod
    # def from_pretrained(cls, model_type, override_args=None):
    #     assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
    #     override_args = override_args or {} # default to empty dict
    #     # only dropout can be overridden see more notes below
    #     assert all(k == 'dropout' for k in override_args)
    #     from transformers import GPT2LMHeadModel
    #     print("loading weights from pretrained gpt: %s" % model_type)

    #     # n_layer, n_head and n_embd are determined from model_type
    #     config_args = {
    #         'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
    #         'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
    #         'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
    #         'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
    #     }[model_type]
    #     print("forcing vocab_size=50257, block_size=1024, bias=True")
    #     config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
    #     config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
    #     config_args['bias'] = True # always True for GPT model checkpoints
    #     # we can override the dropout rate, if desired
    #     if 'dropout' in override_args:
    #         print(f"overriding dropout rate to {override_args['dropout']}")
    #         config_args['dropout'] = override_args['dropout']
    #     # create a from-scratch initialized minGPT model
    #     config = GPTConfig(**config_args)
    #     model = GPT(config)
    #     sd = model.state_dict()
    #     sd_keys = sd.keys()
    #     sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

    #     # init a huggingface/transformers model
    #     model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    #     sd_hf = model_hf.state_dict()

    #     # copy while ensuring all of the parameters are aligned and match in names and shapes
    #     sd_keys_hf = sd_hf.keys()
    #     sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
    #     sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
    #     transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
    #     # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
    #     # this means that we have to transpose these weights when we import them
    #     assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
    #     for k in sd_keys_hf:
    #         if any(k.endswith(w) for w in transposed):
    #             # special treatment for the Conv1D weights we need to transpose
    #             assert sd_hf[k].shape[::-1] == sd[k].shape
    #             with torch.no_grad():
    #                 sd[k].copy_(sd_hf[k].t())
    #         else:
    #             # vanilla copy over the other parameters
    #             assert sd_hf[k].shape == sd[k].shape
    #             with torch.no_grad():
    #                 sd[k].copy_(sd_hf[k])

    #     return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args) #如此华丽炫技的操作
        print(f"using fused AdamW: {use_fused}")

        return optimizer
#mfu指标

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.hidden_size//cfg.num_attention_heads, cfg.max_position_embeddings #cfg.block_size为一个句子的长度
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter  #fwdbwd_per_iter为batch_size
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        # flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        # flops_promised=40e12# 4060laptop bfloat16 peak flops is 40 TFLOPS
        flops_promised=71e12 #3090
        # flops_promised=330e12 #4090
        mfu = flops_achieved / flops_promised
        return mfu
    
#对输入的文本，根据训练好的网络，打印max_new_tokens个新的字符
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,top_p=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.max_position_embeddings else idx[:, -self.config.max_position_embeddings:]
            # forward the model to get the logits for the index in the sequence
            # 这里没有传入target,则只对最后一个词做预测
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # logits [batch,vocab_size]
            # optionally crop the logits to only the top k options
            if top_k is not None:
                #v [Batch, top_k]
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply top_p
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                sorted_probs=F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0 

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                
                
                logits[indices_to_remove] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
