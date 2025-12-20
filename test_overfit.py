'''
Overfit Single Batch测试
为了验证自我搭建的EasyGPT模型代码逻辑正确，可以通过一个简单的“过拟合单批次数据”的测试。
即构建一个非常小的数据集（单个Batch），然后用一个较大的学习率训练模型。
如果模型和训练代码的逻辑没有问题，Loss应该会迅速下降到非常低（接近0），并且模型能够完美记忆这个批次的数据。
下面的代码实现了这个测试过程。
'''
import torch
import time
from model import GPT, GPTConfig

# -----------------------------------------------------------------------------
# 1. 配置参数 
# -----------------------------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on {device}")


config = GPTConfig(
    max_position_embeddings = 64,      
    vocab_size = 50304,  
    num_hidden_layers = 4,          
    num_attention_heads = 4,
    num_key_value_heads=4,
    hidden_size = 128,         
    dropout = 0.0,        
    bias = False,
    head_dim=32,
    intermediate_size=256,
)

#-----------------------------------------------------------------------------
# 2. 初始化模型
# -----------------------------------------------------------------------------
torch.manual_seed(45) # 固定随机种子
model = GPT(config)
model.to(device)
print("Model initialized. Parameters:", sum(p.numel() for p in model.parameters()))

# -----------------------------------------------------------------------------
# 3. 伪造单批次数据
# -----------------------------------------------------------------------------
batch_size = 4
seq_len = config.max_position_embeddings

# 随机生成一些整数当作 token，模拟一个 Batch 的数据
# 形状: [batch_size, seq_len + 1] (+1 是为了切分输入和目标)
data = torch.randint(0, config.vocab_size, (batch_size, seq_len + 1)).to(device)

# 构造输入 X 和 目标 Y

X = data[:, :-1].contiguous()
Y = data[:, 1:].contiguous()

print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")

# -----------------------------------------------------------------------------
# 4. 暴力训练循环
# -----------------------------------------------------------------------------
# 使用较大的学习率 (1e-3)，为了快速收敛
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

print("\nStarting overfitting test...")
start_time = time.time()

# 通常跑 100-500 步足以让 Loss 归零
max_iters = 500

for i in range(max_iters):
    # 1. 前向传播
    logits, loss = model(X, Y)

    # 2. 反向传播
    model.zero_grad(set_to_none=True)
    loss.backward()
    
    # 3. 更新参数
    optimizer.step()

    # 4. 打印日志
    if i % 10 == 0:
        print(f"Iter {i:03d} | Loss: {loss.item():.6f}")

    # 5. 提前终止条件：Loss 足够小说明成功了
    if loss.item() < 0.001:
        print(f"\n[SUCCESS] Loss dropped to {loss.item():.6f} at iter {i}!")
        print("Your model architecture and forward/backward pass are correct.")
        break

total_time = time.time() - start_time
print(f"Finished in {total_time:.2f} seconds.")

# -----------------------------------------------------------------------------
# 5. 最终验证 
# -----------------------------------------------------------------------------
print("\nVerifying predictions...")
model.eval()
with torch.no_grad():
    # 再次输入 X，看看模型预测的 logits 是否和 Y 一致
    logits, _ = model(X, Y) 
    # 取概率最大的 token
    probs = torch.softmax(logits, dim=-1)
    pred_tokens = torch.argmax(probs, dim=-1)

    print("Target (Y):", Y.tolist(), "...") # 打印第一条数据的前10个
    print("Preds  (P):", pred_tokens.tolist(), "...")

    if torch.equal(Y, pred_tokens):
        print("\n[PERFECT MATCH] The model perfectly memorized the batch.")
    else:
        # 只要 loss 很低，偶尔错几个也没事
        print("\n[Match check] Almost perfect match (Loss is low enough).")