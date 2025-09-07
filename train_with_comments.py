"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""
# 脚本目的：训练 nanoGPT 模型，支持单 GPU（GTX 1050 适用）和分布式数据并行（DDP，多 GPU/多节点）。
# - README 关联：对应 quick start（Shakespeare 示例）和 reproducing GPT-2 部分。
# - 单 GPU 示例：`python train.py --batch_size=32 --compile=False`（GTX 1050 需调整 batch_size=4，block_size=64）。
# - DDP 示例：多 GPU/节点设置，GTX 1050 不适用（4GB VRAM，单卡运行）。
# - 注释说明：默认配置针对 GPT-2 124M（n_layer=12，n_embd=768），对 GTX 1050 需大幅缩小。

import os
# 导入 os 模块，用于文件路径操作（如创建输出目录、加载数据集）。

import time
# 导入 time 模块，记录训练时间（如每步耗时、MFU 计算）。

import math
# 导入 math 模块，用于余弦学习率衰减（cosine decay）。

import pickle
# 导入 pickle 模块，加载 meta.pkl（字符映射，来自 prepare.py）。

from contextlib import nullcontext
# 导入 nullcontext，用于非 GPU 环境（CPU）或无混合精度训练时的上下文管理。

import numpy as np
# 导入 numpy，用于数据加载（np.memmap 读取 train.bin/val.bin）。

import torch
# 导入 PyTorch，核心深度学习框架，支持 GPU 加速（GTX 1050 需 CUDA 11.8）。

from torch.nn.parallel import DistributedDataParallel as DDP
# 导入 DDP 模块，支持分布式训练（GTX 1050 不适用，单卡运行）。

from torch.distributed import init_process_group, destroy_process_group
# 导入分布式训练初始化/清理函数，仅 DDP 使用。

from model import GPTConfig, GPT
# 导入 model.py 的 GPTConfig（模型配置）和 GPT（模型定义）。
# - GPTConfig：定义模型参数（n_layer、n_head、n_embd 等）。
# - GPT：实现 GPT 模型，支持从头训练或加载 GPT-2 权重。

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# 默认配置：针对 GPT-2 124M（12 层，768 嵌入维度）在 OpenWebText 数据集上训练。
# - GTX 1050 需调整：n_layer=4，n_embd=128，batch_size=4，block_size=64（参考 README quick start）。

# I/O
out_dir = 'out'
# 输出目录：保存 checkpoint（ckpt.pt）和日志，默认 'out'。
# - GTX 1050 建议：设为 'out-shakespeare-1050'，避免覆盖。

eval_interval = 100
# 评估间隔：每 2000 步在 train/val 上评估 loss。
# - GTX 1050 建议：设为 100（小模型，快速验证）。

log_interval = 10
# 日志间隔：每 1 步打印 loss 和时间。
# - GTX 1050 建议：设为 10，减少 I/O 开销。

eval_iters = 10
# 评估时迭代次数：每次评估计算 x 个 batch 的平均 loss。
# - GTX 1050 建议：设为 10，减少计算量（4GB VRAM 限制）。

eval_only = False # if True, script exits right after the first eval
# 仅评估模式：若 True，仅运行一次评估后退出。
# - 调试用，GTX 1050 可设 True 快速测试环境。

always_save_checkpoint = True # if True, always save a checkpoint after each eval
# 是否每次评估都保存 checkpoint（即使 loss 未改善）。
# - GTX 1050 建议：保持 True，磁盘空间足够（checkpoint ~10-50MB）。

init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# 模型初始化方式：
# - 'scratch'：从头训练（GTX 1050 适用，Shakespeare 示例）。
# - 'resume'：从 checkpoint 继续训练。
# - 'gpt2*'：加载 OpenAI GPT-2 权重（如 gpt2，124M；GTX 1050 需 batch_size=1-2）。
# - GTX 1050 建议：用 'scratch' 或 'gpt2'（小模型）。

# wandb logging
wandb_log = False # disabled by default
# 是否启用 Weights & Biases 日志（在线记录 loss、lr 等）。
# - GTX 1050 建议：保持 False，节省资源（无需联网）。

wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# wandb 项目名和运行名，默认针对 OpenWebText 和 GPT-2。
# - GTX 1050 建议：若启用，设为 'shakespeare-1050'。

# data
dataset = 'openwebtext'
# 数据集名称：默认 OpenWebText（大，几十 GB）。
# - GTX 1050 必须改：设为 'shakespeare_char'（1MB，参考 README quick start）。

gradient_accumulation_steps = 4 # used to simulate larger batch sizes
# 梯度累积步数：模拟大 batch size（5*8=40，针对 8 GPU）。
# - GTX 1050 建议：设为 1-4（单 GPU，4GB VRAM 限制）。

batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
# 微批次大小：每个 GPU 的 batch size（DDP 下）。
# - GTX 1050 建议：设为 2-4（避免 OOM，参考 README CPU 示例）。

block_size = 128
# 上下文长度：每个样本的最大 token 数（GPT-2 默认 1024）。
# - GTX 1050 建议：设为 64-128（4GB VRAM 限制）。

# model
n_layer = 4
n_head = 2
n_embd = 128
# 模型参数：12 层，12 注意力头，768 嵌入维度（GPT-2 124M 配置）。
# - GTX 1050 建议：n_layer=4，n_head=4，n_embd=128（约 10M 参数，参考 README quick start）。

dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
# Dropout 比例：预训练用 0，微调用 0.1+。
# - GTX 1050 建议：设为 0（小模型，减少正则化）。

bias = False # do we use bias inside LayerNorm and Linear layers?
# 是否在 LayerNorm 和 Linear 层中使用 bias。
# - 默认 False（GPT-2 风格），GTX 1050 无需改。

# adamw optimizer
learning_rate = 6e-4 # max learning rate
# 最大学习率：AdamW 优化器。
# - GTX 1050 建议：保持 6e-4（Shakespeare 小模型适用）。

max_iters = 600000 # total number of training iterations
# 最大训练迭代次数：针对 GPT-2 124M（4 天，8x A100）。
# - GTX 1050 建议：设为 500-2000（1-2 小时，参考 README CPU 示例）。

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
# AdamW 参数：权重衰减和动量。
# - GTX 1050 建议：保持默认（小模型稳定）。

grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# 梯度裁剪：限制梯度范数，防止爆炸。
# - GTX 1050 建议：保持 1.0（稳定训练）。

# learning rate decay settings
decay_lr = True # whether to decay the learning rate 是否降低学习率
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla 最小学习率
# 学习率衰减：使用余弦衰减（warmup + cosine decay）。
# - GTX 1050 建议：warmup_iters=100，lr_decay_iters=500-2000（匹配 max_iters）。

# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# 分布式训练后端：nccl 适合 GPU。
# - GTX 1050：单卡运行，无需 DDP，忽略此项。

# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
# 设备：默认 cuda（GPU）。
# - GTX 1050 必须设：--device=cuda（已验证 torch.cuda.is_available() == True）。

dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16'
# 数据类型：bfloat16（需 Ampere+，1050 不支持）或 float16（需 GradScaler）。
# - GTX 1050 建议：设为 float16（Pascal 支持，节省 VRAM）。

compile = True # use PyTorch 2.0 to compile the model to be faster
# 是否使用 torch.compile（PyTorch 2.0 加速）。
# - GTX 1050 必须设：--compile=False（Pascal 不稳定，README troubleshooting 提及）。

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# 收集配置变量（int/float/bool/str 类型的全局变量），排除私有变量（_ 开头）。

exec(open('configurator.py').read()) # overrides from command line or config file
# 执行 configurator.py，允许命令行或配置文件覆盖默认参数。
# - 例：python train.py --batch_size=4 覆盖 batch_size。
# - GTX 1050 建议：用 config/train_shakespeare_char.py 或命令行指定参数。

config = {k: globals()[k] for k in config_keys} # will be useful for logging
# 创建配置字典，保存所有配置参数，用于日志和 checkpoint。

# -----------------------------------------------------------------------------
# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
# 检查是否为 DDP 运行（通过环境变量 RANK）。
# - GTX 1050：单卡，ddp=False，忽略 DDP 相关代码。

if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
# DDP 初始化：设置进程组、设备、种子等。
# - GTX 1050：单卡运行，跳过此块。

else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
# 单 GPU 设置：主进程，种子偏移为 0，世界大小为 1。
# - GTX 1050：适用，单卡训练。

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
# 计算每迭代的 token 数：梯度累积 × 进程数 × batch_size × block_size。
# - GTX 1050 示例：1 * 1 * 4 * 64 = 256 tokens/iter。

print(f"tokens per iteration will be: {tokens_per_iter:,}")
# 打印每迭代 token 数，格式化带千位分隔符。

if master_process:
    os.makedirs(out_dir, exist_ok=True)
# 主进程创建输出目录（out_dir，如 'out-shakespeare-1050'）。
# - exist_ok=True：目录存在不报错。

torch.manual_seed(1337 + seed_offset)
# 设置随机种子（1337 + 偏移），确保可重复性。
# - GTX 1050：seed_offset=0（单卡）。

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
# 启用 TF32（TensorFloat-32）加速矩阵运算（仅 Ampere+ 支持）。
# - GTX 1050：Pascal 不支持 TF32，但无害，保持默认。

device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# 设置设备类型：cuda 或 cpu。
# - GTX 1050：device_type='cuda'。

# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
# 映射数据类型：float16 触发 GradScaler。
# - GTX 1050：ptdtype=torch.float16（节省 VRAM）。

ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
# 设置混合精度上下文：CPU 用 nullcontext，GPU 用 autocast（float16）。
# - GTX 1050：启用 autocast，优化性能/VRAM。

# poor man's data loader
data_dir = os.path.join('data', dataset)
# 数据目录：如 data/shakespeare_char（GTX 1050 用）。

def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    # 加载数据集：train.bin 或 val.bin（来自 prepare.py，uint16 格式）。
    # - np.memmap：内存映射，高效读取大文件（Shakespeare ~2MB）。
    # - GTX 1050：数据小，I/O 非瓶颈。

    ix = torch.randint(len(data) - block_size, (batch_size,))
    # 随机采样 batch_size 个起始索引（0 到 len(data)-block_size）。
    # - 确保每个样本长度为 block_size。

    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    # 生成输入 x（block_size 个 token）和目标 y（x 的后一位，偏移 1）。
    # - 例：x=[1,2,3]，y=[2,3,4]（语言模型预测下一 token）。
    # - astype(np.int64)：转为 int64，兼容 PyTorch。

    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    # 移动数据到设备（cuda 或 cpu）。
    # - pin_memory：加速 CPU 到 GPU 数据传输。
    # - GTX 1050：device='cuda'，异步传输减少延迟。

    return x, y
# 返回批次数据：x（输入），y（目标）。

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9
# 初始化迭代计数和最佳验证 loss（初始为无穷大）。

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
# 加载 meta.pkl（来自 prepare.py），获取词汇表大小（Shakespeare: 65）。
# - GTX 1050：meta_vocab_size=65（字符级，小词汇表适合低资源）。

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
# 初始化模型参数字典，传递给 GPTConfig。

if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    # 设置词汇表大小：优先用 meta.pkl 的（65），否则默认 GPT-2 的 50304。
    # - GTX 1050：用 meta_vocab_size=65（Shakespeare）。

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
# 从头初始化模型：创建 GPTConfig 和 GPT 实例。
# - GTX 1050 建议：n_layer=4，n_embd=128，block_size=64。

elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
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
# 从 checkpoint 恢复训练：加载模型参数、优化器状态等。
# - GTX 1050：若有 checkpoint（如 out-shakespeare-1050/ckpt.pt），可继续训练。

elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# 加载 GPT-2 预训练权重（如 gpt2，124M）。
# - GTX 1050：仅 gpt2（124M）可行，需 batch_size=1-2，block_size=128。

# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
# 调整模型 block_size（若命令行 block_size 小于模型默认）。
# - GTX 1050：block_size=64，确保 VRAM 安全。

model.to(device)
# 移动模型到设备（cuda）。
# - GTX 1050：device='cuda'。

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
# 初始化 GradScaler：float16 训练时防止梯度下溢。
# - GTX 1050：dtype='float16'，启用 scaler。

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
# 配置 AdamW 优化器（由 model.py 定义）。
# - GTX 1050：参数合适（learning_rate=6e-4，weight_decay=1e-1）。

if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
# 恢复优化器状态（若 resume）。

checkpoint = None # free up memory
# 释放 checkpoint 内存。

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0
# 编译模型（PyTorch 2.0 加速）。
# - GTX 1050 必须：compile=False（Pascal 不稳定）。

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
# 包装为 DDP 模型（多 GPU）。
# - GTX 1050：ddp=False，跳过。

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
# 评估函数：计算 train/val 的平均 loss（eval_iters 个 batch）。
# - torch.no_grad()：禁用梯度计算，节省内存。
# - GTX 1050：eval_iters=10，减少评估时间。

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
# 学习率调度：线性 warmup + 余弦衰减。
# - GTX 1050：warmup_iters=100，lr_decay_iters=500，匹配 max_iters。

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)
# 初始化 wandb 日志（记录 loss、lr、MFU）。
# - GTX 1050：wandb_log=False，节省资源。

# training loop
X, Y = get_batch('train') # fetch the very first batch
# 获取第一个训练 batch。

t0 = time.time()
# 记录开始时间。

local_iter_num = 0 # number of iterations in the lifetime of this process
# 本地迭代计数（单进程用）。

raw_model = model.module if ddp else model # unwrap DDP container if needed
# 获取原始模型（DDP 解包）。
# - GTX 1050：raw_model=model（无 DDP）。

running_mfu = -1.0
# 初始化 MFU（Model FLOPs Utilization，模型计算效率）。

while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # 设置当前迭代学习率（余弦衰减或固定）。

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    # 定期评估 loss，保存 checkpoint（若 val loss 改善或 always_save_checkpoint）。
    # - GTX 1050：eval_interval=100，checkpoint 小（~10-50MB）。

    if iter_num == 0 and eval_only:
        break
    # 若 eval_only=True，评估一次后退出。

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # 前向传播：计算 logits 和 loss，loss 除以梯度累积步数。
        # - GTX 1050：gradient_accumulation_steps=1-4。

        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # 异步预取下一 batch，减少 GPU 空闲时间。

        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
        # 反向传播：缩放 loss（float16），计算梯度。

    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # 梯度裁剪：限制梯度范数。
    # - GTX 1050：grad_clip=1.0，稳定训练。

    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # 更新优化器和 scaler（float16 训练）。

    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)
    # 清零梯度，释放内存。

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    # 日志：打印迭代 loss、时间、MFU（计算效率）。
    # - GTX 1050：预期 100-500ms/iter，MFU 低（~5-10%）。

    iter_num += 1
    local_iter_num += 1
    # 更新迭代计数。

    # termination conditions
    if iter_num > max_iters:
        break
# 终止条件：达到 max_iters（GTX 1050：500-2000）。

if ddp:
    destroy_process_group()
# 清理 DDP 进程组（GTX 1050 忽略）。

# GTX 1050 推荐命令（参考 README quick start 和低资源调整）：
# python train.py config/train_shakespeare_char.py --device=cuda --compile=False --eval_iters=10 --log_interval=10 --block_size=64 --batch_size=4 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=500 --lr_decay_iters=500 --dropout=0.0 --out_dir=out-shakespeare-1050
# - 运行后生成 checkpoint（out-shakespeare-1050/ckpt.pt），用于 sample.py。
# - 预期：1-2 小时，val loss ~1.5-1.8（参考 README Shakespeare 示例）。