## <center>EasyGPT</center>
---
- 基于对karpathy的[nanoGPT](https://github.com/karpathy/nanoGPT?tab=readme-ov-file)项目的学习,加入了自己的理解和一些更新的模块,实现了在单张RTX3090上预训练303m大小的GPT模型(90小时),训练过程中显存占用约为15GB.

### Hardware
- 1X RTX3090

### Model Config
- 采用24层transformer结构,改进了原GPT-2,加入了较新的RMSNorm,RotaryEmbedding,SwiGLU,GQA结构,总参数量为303m.
```python
    max_position_embeddings: int = 1024
    vocab_size: int = 50304
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads:int=4
    head_dim:int=64
    dropout: float = 0.0
    bias: bool = False 
    intermediate_size:int=2560
    norm_eps:float=1e-6
    rope_theta:int=10000
```
### Install
- refer requirements.txt
```bash
pip install -r requirements.txt
```
### Quick Start
```bash
python train.py config/train_gpt.py
```
### Datasets
- [Skylion007/openwebtext](https://huggingface.co/datasets/Skylion007/openwebtext/tree/main)
- 若网速不佳可以考虑使用镜像[hf-mirror](https://hf-mirror.com/)
- 建议先下载至本地再load_dataset
- 可参考如下步骤(支持断点续下载)
```bash
pip install huggingface-hub==0.23.4

export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download \
  Skylion007/openwebtext \
  --repo-type dataset \
  --resume-download \
  --local-dir openwebtext

python data/prepare.py
```
完成数据集(包含20个tar文件)下载后运行prepare.py即可得到train.bin(17GB)和val.bin,注意磁盘应预留约100GB空间
- 如下载遇困难可通过我的[百度网盘分享]()直接下载train.bin和val.bin

### Outcomes
- 每个step为0.5m tokens,共运行15k步(成本有限),耗时90h,也可考虑将模型层数适当改小至12层.
### Reference
[nanoGPT](https://github.com/karpathy/nanoGPT?tab=readme-ov-file)
