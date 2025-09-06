"""
A much shorter version of train.py for benchmarking
"""
# 脚本目的：train.py 的简化版，用于基准测试 nanoGPT 模型性能（时间、MFU）。
# - README 关联：对应 efficiency notes 部分，用于简单模型基准测试和分析，忽略训练复杂性（如评估、日志）。
# - GTX 1050 适用：需调整 batch_size=4，block_size=64，n_layer=4，n_embd=128 以适应 4GB VRAM；--compile=False（Pascal 不稳定）；--device=cuda。
# - 运行示例：python bench.py --batch_size=4 --block_size=64 --device=cuda --compile=False --profile=False --real_data=True
# - 输出：迭代时间（ms/iter）和 MFU（%），评估模型效率。

import os
# 导入 os 模块，用于文件路径操作（如数据目录）。

from contextlib import nullcontext
# 导入 nullcontext，用于非混合精度上下文（CPU 或 float32）。

import numpy as np
# 导入 numpy，用于数据加载（np.memmap 读取 train.bin）。

import time
# 导入 time 模块，记录基准测试时间（dt 计算）。

import torch
# 导入 PyTorch，核心框架，支持 GPU 基准（GTX 1050 需 CUDA 11.8）。

from model import GPTConfig, GPT
# 导入 model.py 的 GPTConfig（配置）和 GPT（模型）。

# -----------------------------------------------------------------------------
batch_size = 12
# 批次大小：默认 12（微批次）。
# - GTX 1050 建议：设为 2-4（避免 OOM）。

block_size = 1024
# 上下文长度：默认 1024（GPT-2）。
# - GTX 1050 建议：设为 64-128（减少 VRAM）。

bias = False
# 是否在 Linear 和 LayerNorm 使用 bias：默认 False（简化模型）。

real_data = True
# 是否使用真实数据：True 用 OpenWebText，False 用随机数据。
# - GTX 1050 建议：real_data=True，用 Shakespeare（dataset='shakespeare_char'）。

seed = 1337
# 随机种子：确保可重复性。

device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
# 设备：默认 cuda（GPU）。
# - GTX 1050：--device=cuda（确认 torch.cuda.is_available() == True）。

dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
# 数据类型：bfloat16（Ampere+）或 float16。
# - GTX 1050：float16（Pascal 支持，节省 VRAM）。

compile = True # use PyTorch 2.0 to compile the model to be faster
# 是否编译模型（PyTorch 2.0 加速）。
# - GTX 1050：--compile=False（Pascal 不稳定，README troubleshooting）。

profile = False # use pytorch profiler, or just simple benchmarking?
# 是否使用 PyTorch Profiler：False 用简单基准，True 用详细分析（输出到 ./bench_log）。
# - GTX 1050 建议：profile=False（简单基准足够）。

exec(open('configurator.py').read()) # overrides from command line or config file
# 执行 configurator.py，允许命令行覆盖参数（如 --batch_size=4）。
# - GTX 1050：用命令行调整参数以优化。

# -----------------------------------------------------------------------------
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# 设置随机种子（CPU 和 GPU），确保一致性。

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
# 启用 TF32 加速（Ampere+ 支持）。
# - GTX 1050：Pascal 不支持，但无害。

device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# 设置设备类型：cuda 或 cpu。

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
# 映射数据类型。

ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
# 设置混合精度上下文：CPU 用 nullcontext，GPU 用 autocast（float16/bfloat16）。

# data loading init
if real_data:
    dataset = 'openwebtext'
    # 数据集：默认 OpenWebText。
    # - GTX 1050 建议：dataset='shakespeare_char'（小数据集，1MB）。
    data_dir = os.path.join('data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    # 加载 train.bin（uint16 格式，内存映射）。
    def get_batch(split):
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        data = train_data # note ignore split in benchmarking script
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x, y
    # 获取 batch 函数：随机采样输入 x 和目标 y（忽略 split，只用 train_data）。
    # - GTX 1050：batch_size=4，block_size=64，异步传输减少延迟。
else:
    # alternatively, if fixed data is desired to not care about data loading
    x = torch.randint(50304, (batch_size, block_size), device=device)
    y = torch.randint(50304, (batch_size, block_size), device=device)
    get_batch = lambda split: (x, y)
    # 随机数据模式：固定随机 tensor，忽略数据加载（纯测试计算速度）。

# model init
gptconf = GPTConfig(
    block_size = block_size, # how far back does the model look? i.e. context size
    n_layer = 12, n_head = 12, n_embd = 768, # size of the model
    dropout = 0, # for determinism
    bias = bias,
)
# 模型配置：默认 GPT-2 124M（12 层，768 嵌入）。
# - GTX 1050 建议：n_layer=4，n_head=4，n_embd=128（小模型）。

model = GPT(gptconf)
# 创建 GPT 模型实例。

model.to(device)
# 移动模型到设备（cuda）。

optimizer = model.configure_optimizers(weight_decay=1e-2, learning_rate=1e-4, betas=(0.9, 0.95), device_type=device_type)
# 配置 AdamW 优化器（简化参数）。

if compile:
    print("Compiling model...")
    model = torch.compile(model) # pytorch 2.0
# 编译模型（加速）。
# - GTX 1050：compile=False。

if profile:
    # useful docs on pytorch profiler:
    # - tutorial https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
    # - api https://pytorch.org/docs/stable/profiler.html#torch.profiler.profile
    wait, warmup, active = 5, 5, 5
    # Profiler 阶段：wait=5（跳过），warmup=5（预热），active=5（记录）。
    num_steps = wait + warmup + active
    # 总步数：15。
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./bench_log'),
        record_shapes=False,
        profile_memory=False,
        with_stack=False, # incurs an additional overhead, disable if not needed
        with_flops=True,
        with_modules=False, # only for torchscript models atm
    ) as prof:
        # 初始化 Profiler：记录 CPU/CUDA 活动，输出到 ./bench_log（TensorBoard 格式）。

        X, Y = get_batch('train')
        # 获取第一个 batch。
        for k in range(num_steps):
            with ctx:
                logits, loss = model(X, Y)
            # 前向传播。
            X, Y = get_batch('train')
            # 预取下一 batch。
            optimizer.zero_grad(set_to_none=True)
            # 清零梯度。
            loss.backward()
            # 反向传播。
            optimizer.step()
            # 优化器步进。
            lossf = loss.item()
            print(f"{k}/{num_steps} loss: {lossf:.4f}")

            prof.step() # notify the profiler at end of each step
        # 通知 Profiler 步进结束。
        # - GTX 1050：profile=False（简单基准足够），若用，检查 VRAM。

else:
    # simple benchmarking
    # 简单基准模式。
    torch.cuda.synchronize()
    # 同步 GPU，确保时间准确。
    for stage, num_steps in enumerate([10, 20]): # burnin, then benchmark
        # 两个阶段：burnin（10 步，预热），benchmark（20 步，测量）。
        t0 = time.time()
        # 记录开始时间。
        X, Y = get_batch('train')
        for k in range(num_steps):
            with ctx:
                logits, loss = model(X, Y)
            X, Y = get_batch('train')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            lossf = loss.item()
            print(f"{k}/{num_steps} loss: {lossf:.4f}")
        # 训练循环：前向 + 反向 + 更新。
        torch.cuda.synchronize()
        # 同步结束时间。
        t1 = time.time()
        dt = t1-t0
        # 计算总时间。
        mfu = model.estimate_mfu(batch_size * 1 * num_steps, dt)
        # 估算 MFU（单进程，fwdbwd_per_iter=batch_size * num_steps）。
        if stage == 1:
            print(f"time per iteration: {dt/num_steps*1000:.4f}ms, MFU: {mfu*100:.2f}%")
        # 仅 benchmark 阶段打印：ms/iter 和 MFU（%）。
        # - GTX 1050 预期：100-500ms/iter，MFU ~5-10%（远低于 A100 的 312 TFLOPS 基准）。