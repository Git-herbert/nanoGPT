"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""
# 脚本目的：定义 nanoGPT 的核心模型结构，包括 GPT 模型、注意力机制、MLP 等。
# - README 关联：对应 `model.py`（~300 行），实现 GPT 模型，支持从头训练（Shakespeare 示例）或加载 GPT-2 权重。
# - GTX 1050 适用：需调整参数（如 n_layer=4，n_embd=128，block_size=64）以适应 4GB VRAM。
# - 参考：基于 OpenAI GPT-2 和 Hugging Face 的实现，简化为教学/研究用途。

import math
# 导入 math 模块，用于注意力机制中的缩放因子（1/sqrt(d_k)）等计算。

import inspect
# 导入 inspect 模块，检查 AdamW 优化器是否支持 fused 参数。

from dataclasses import dataclass
# 导入 dataclass，用于定义 GPTConfig（配置类，简化参数管理）。

import torch
# 导入 PyTorch，核心深度学习框架，支持 GPU 加速（GTX 1050 需 CUDA 11.8）。

import torch.nn as nn
# 导入 PyTorch 神经网络模块（nn.Module、nn.Linear 等）。

from torch.nn import functional as F
# 导入 PyTorch 函数式模块（F.softmax、F.layer_norm 等）。

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    # 自定义 LayerNorm 模块，支持可选 bias（PyTorch 的 F.layer_norm 不支持 bias=False）。

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        # 初始化权重为全 1（维度 ndim，n_embd）。
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        # 初始化偏置为全 0（若 bias=True），否则 None。

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        # 前向传播：应用 LayerNorm，标准化输入（均值 0，方差 1）。
        # - input：输入张量（B, T, C）。
        # - weight/bias：缩放/偏移参数。
        # - eps=1e-5：避免除零。

class CausalSelfAttention(nn.Module):
    # 因果自注意力模块（单层 Transformer 的核心组件）。

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # 确保嵌入维度（n_embd）可被头数（n_head）整除，每头维度 hs = n_embd / n_head。
        # - GTX 1050 建议：n_embd=128，n_head=4（每头 32）。

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # 线性层：将输入映射为 Q、K、V（3*n_embd，批量计算所有头）。
        # - 输入：(B, T, n_embd)，输出：(B, T, 3*n_embd)。

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # 输出投影：将多头注意力结果映射回 n_embd 维度。

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # Dropout：注意力矩阵和输出投影的正则化。
        # - GTX 1050 建议：dropout=0（小模型减少正则化）。

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # 保存配置参数：头数、嵌入维度、Dropout 率。

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        # 检查是否支持 Flash Attention（PyTorch 2.0+，高效注意力机制）。
        # - GTX 1050：Pascal 不支持 Flash Attention（需 PyTorch 2.0 和 Ampere+），用慢速注意力。

        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # 警告：若无 Flash Attention，使用手动实现（慢但兼容）。
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
            # 因果掩码：下三角矩阵（1 表示可见，0 表示屏蔽），确保只关注左侧 token。
            # - 形状：(1, 1, block_size, block_size)。
            # - GTX 1050：block_size=64，掩码小，内存占用低。

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # 输入 x 形状：(B, T, C)，B=batch_size，T=block_size，C=n_embd。

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # 计算 Q、K、V：通过 c_attn 线性层，输出 (B, T, 3*n_embd)，拆分为三个 (B, T, n_embd)。
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # 重塑 Q、K、V 为多头格式：(B, T, n_head, hs) -> (B, n_head, T, hs)，hs=n_embd/n_head。

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
            # Flash Attention：高效计算因果注意力（PyTorch 2.0+）。
            # - is_causal=True：自动应用因果掩码。
            # - GTX 1050：不适用，flash=False。
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # 手动注意力：计算 QK^T / sqrt(d_k)，形状 (B, nh, T, T)。
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            # 应用因果掩码：屏蔽未来 token（设为 -inf）。
            att = F.softmax(att, dim=-1)
            # Softmax 归一化，得到注意力权重。
            att = self.attn_dropout(att)
            # 应用注意力 Dropout。
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            # 计算加权值：注意力权重 × V。
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # 重塑输出：(B, nh, T, hs) -> (B, T, n_embd)，合并多头结果。

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        # 输出投影：通过 c_proj 线性层，应用 Dropout。
        return y

class MLP(nn.Module):
    # 多层感知器（MLP）模块，Transformer 的前馈网络。

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        # 第一线性层：扩展维度（n_embd -> 4*n_embd）。
        self.gelu = nn.GELU()
        # GELU 激活函数（GPT-2 标准）。
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        # 第二线性层：还原维度（4*n_embd -> n_embd）。
        self.dropout = nn.Dropout(config.dropout)
        # Dropout 正则化。

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
        # 前向传播：线性 -> GELU -> 线性 -> Dropout。

class Block(nn.Module):
    # Transformer 块：LayerNorm + 因果自注意力 + MLP。

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        # 第一个 LayerNorm（前置注意力）。
        self.attn = CausalSelfAttention(config)
        # 因果自注意力模块。
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        # 第二个 LayerNorm（前置 MLP）。
        self.mlp = MLP(config)
        # MLP 模块。

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        # 残差连接：x + 注意力输出（LayerNorm 后）。
        x = x + self.mlp(self.ln_2(x))
        # 残差连接：x + MLP 输出（LayerNorm 后）。
        return x

@dataclass
class GPTConfig:
    # GPT 模型配置类，使用 dataclass 简化参数管理。
    block_size: int = 1024
    # 上下文长度：默认 1024（GPT-2 标准）。
    # - GTX 1050 建议：block_size=64（减少 VRAM）。
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    # 词汇表大小：默认 50304（GPT-2 BPE，填充到 64 倍数）。
    # - GTX 1050：Shakespeare 用 65（来自 meta.pkl）。
    n_layer: int = 12
    # Transformer 层数：默认 12（GPT-2 124M）。
    # - GTX 1050 建议：n_layer=4。
    n_head: int = 12
    # 注意力头数：默认 12。
    # - GTX 1050 建议：n_head=4。
    n_embd: int = 768
    # 嵌入维度：默认 768。
    # - GTX 1050 建议：n_embd=128。
    dropout: float = 0.0
    # Dropout 率：默认 0（预训练）。
    # - GTX 1050 建议：dropout=0。
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    # 是否使用 bias：默认 True（GPT-2 风格）。
    # - GTX 1050：保持 True。

class GPT(nn.Module):
    # GPT 模型主类，整合所有组件。

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        # 确保 vocab_size 和 block_size 已设置。
        self.config = config
        # 保存配置。

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # Token 嵌入：将 token ID 映射为 n_embd 向量。
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # 位置嵌入：将位置 ID（0 到 block_size-1）映射为 n_embd 向量。
            drop = nn.Dropout(config.dropout),
            # 输入 Dropout。
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # Transformer 块列表：n_layer 个 Block。
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
            # 最终 LayerNorm。
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # 语言模型头：将 n_embd 映射到 vocab_size（预测 token 概率）。
        # - bias=False：减少参数。

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying
        # 权重共享：token 嵌入和 lm_head 共享权重（减少参数，GPT-2 标准）。

        # init all weights
        self.apply(self._init_weights)
        # 初始化所有权重（调用 _init_weights）。
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        # 特殊初始化：残差投影（c_proj）权重，标准差缩放（GPT-2 论文推荐）。

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
        # 打印模型参数量（单位：百万）。
        # - GTX 1050 示例：n_layer=4，n_embd=128，vocab_size=65，约 1-2M 参数。

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        # 计算总参数量。
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
            # 排除位置嵌入参数（wpe）。
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # 权重初始化：线性层和嵌入层用正态分布（std=0.02），偏置初始化为 0。

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        # 输入 idx：(B, T)，token ID 序列。
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        # 确保序列长度不超过 block_size。

        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
        # 位置索引：0 到 t-1。

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        # Token 嵌入：将 token ID 转为向量。
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        # 位置嵌入：将位置 ID 转为向量。
        x = self.transformer.drop(tok_emb + pos_emb)
        # 输入：token 嵌入 + 位置嵌入，应用 Dropout。
        for block in self.transformer.h:
            x = block(x)
        # 通过 n_layer 个 Transformer 块。
        x = self.transformer.ln_f(x)
        # 最终 LayerNorm。

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            # 计算 logits：(B, T, vocab_size)，预测每个 token 的概率。
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            # 计算交叉熵损失：展平 logits 和 targets，忽略 -1（填充）。
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            # 推理优化：仅对最后位置计算 logits（节省计算量）。
            loss = None

        return logits, loss
        # 返回：logits（预测概率），loss（训练时损失）。

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        # 裁剪位置嵌入权重（适应新 block_size）。
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]
        # 裁剪注意力掩码（若存在）。

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        # 从 Hugging Face 加载预训练 GPT-2 模型。
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        # 定义预训练模型参数。
        # - GTX 1050：仅 gpt2（124M）可行，需 batch_size=1-2，block_size=128。
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # 设置 GPT-2 默认参数。
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # 允许覆盖 dropout。
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        # 加载 Hugging Face 的 GPT-2 模型。

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # 需要转置的权重（Hugging Face 用 Conv1D，nanoGPT 用 Linear）。
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        # 复制权重，确保形状匹配（转置 Conv1D 权重）。

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # 配置 AdamW 优化器。
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # 获取所有参数。
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # 过滤无需梯度的参数。
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        # 分组：2D 参数（权重、嵌入）应用权重衰减，1D 参数（偏置、LayerNorm）不衰减。
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # 打印参数统计。
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        # 创建 AdamW 优化器，优先用 fused 版本（CUDA 加速）。
        # - GTX 1050：use_fused=True（CUDA 11.8 支持）。

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # 估算模型计算效率（MFU，Model FLOPs Utilization）。
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # 计算每迭代的 FLOPs：基于 PaLM 公式。
        # - GTX 1050 示例：n_layer=4，n_head=4，n_embd=128，block_size=64，参数 ~1-2M，FLOPs 低。
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        # 计算 MFU：实际 FLOPs / A100 峰值（312 TFLOPS）。
        # - GTX 1050：MFU 低（~5-10%，Pascal 性能远低于 A100）。
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        # 生成函数：根据输入 idx 生成 max_new_tokens 个新 token。
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # 裁剪输入：确保不超过 block_size。
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # 前向传播：计算 logits。
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # 取最后位置 logits，除以 temperature（控制生成多样性）。
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # Top-k 采样：保留 top k 个概率最高的 token。
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # Softmax：转为概率分布。
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # 多项采样：随机选择下一 token。
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            # 追加新 token，继续生成。

        return idx
        # 返回生成序列。

# GTX 1050 推荐配置（参考 README quick start 和 train.py）：
# - n_layer=4, n_head=4, n_embd=128, block_size=64, vocab_size=65 (Shakespeare, meta.pkl), dropout=0.
# - 参数量：~1-2M，VRAM < 3.5GB，适合 4GB 限制。
# - 运行示例：python train.py config/train_shakespeare_char.py --device=cuda --compile=False --block_size=64 --batch_size=4 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=500
# - 预期：生成 Shakespeare 风格文本（val loss ~1.5-1.8）。