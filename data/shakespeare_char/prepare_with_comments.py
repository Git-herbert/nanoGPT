"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
# 脚本目的：为 nanoGPT 的字符级语言模型准备 Tiny Shakespeare 数据集。
# - 与 README 的关联：对应 quick start 部分，用于生成 train.bin 和 val.bin 文件，作为训练输入。
# - 字符级建模：不像 GPT-2 使用 BPE（字节对编码），这里直接将字符映射为整数（简单高效，适合小模型和 GTX 1050 低资源场景）。
# - 输出文件：
#   - train.bin：训练集的整数编码（uint16 格式）。
#   - val.bin：验证集的整数编码。
#   - meta.pkl：保存字符到整数的映射（stoi）和反向映射（itos）及词汇表大小。

import os
# 导入 os 模块，用于文件路径操作（如拼接目录、检查文件存在）。

import pickle
# 导入 pickle 模块，用于序列化 meta 数据（如字符映射）到 meta.pkl 文件。

import requests
# 导入 requests 模块，用于从 URL 下载 Shakespeare 数据集。

import numpy as np
# 导入 numpy，用于高效数组操作（如将字符 ID 转为 uint16 并保存为二进制文件）。

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
# 定义输入文件路径：input.txt，位于脚本同级目录（data/shakespeare_char/）。
# - os.path.dirname(__file__)：获取当前脚本所在目录。
# - os.path.join：跨平台拼接路径（如 Linux 的 / 和 Windows 的 \）。

if not os.path.exists(input_file_path):
    # 检查 input.txt 是否存在，若不存在则下载。
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    # 数据源：Tiny Shakespeare 数据集（约 1MB），来自 karpathy/char-rnn 项目。
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)
    # 使用 requests.get 下载文本内容，写入 input.txt。
    # - requests.get(data_url).text：获取 URL 的文本内容（Shakespeare 剧本）。
    # - open(..., 'w')：以写模式创建/覆盖 input.txt 文件。

with open(input_file_path, 'r') as f:
    data = f.read()
# 读取 input.txt 内容到 data 变量（字符串类型）。
# - open(..., 'r')：以读模式打开文件。
# - data 包含整个 Shakespeare 数据集的原始文本（约 1,115,394 字符，注释中有输出）。

print(f"length of dataset in characters: {len(data):,}")
# 打印数据集字符总数（约 1,115,394）。
# - len(data)：字符串长度。
# - :, 格式化：添加千位分隔符，便于阅读。

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
# 获取文本中所有唯一字符，排序后存入 chars 列表。
# - set(data)：提取唯一字符（去重）。
# - list(set(data))：转为列表。
# - sorted(...)：按字符顺序排序（如 !, A, a）。
# - 结果：65 个唯一字符（包括字母、标点、空格、换行等）。

vocab_size = len(chars)
# 计算词汇表大小（vocab_size = 65，唯一字符数）。
# - 在字符级建模中，词汇表小（对比 GPT-2 的 BPE 词汇表 ~50,000），适合 GTX 1050 小模型训练。

print("all the unique characters:", ''.join(chars))
# 打印所有唯一字符，合并为字符串（如 "!$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"）。
# - ''.join(chars)：将字符列表拼接成单个字符串。

print(f"vocab size: {vocab_size:,}")
# 打印词汇表大小（65），格式化带千位分隔符。

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
# 创建字符到整数的映射（stoi: string to integer）。
# - enumerate(chars)：为每个字符分配索引（0 到 64）。
# - 例：stoi['a'] = 0, stoi['b'] = 1, ..., stoi['z'] = 25。
# - 字典推导式：{char: index}。

itos = { i:ch for i,ch in enumerate(chars) }
# 创建整数到字符的映射（itos: integer to string）。
# - 反向映射，例：itos[0] = 'a', itos[1] = 'b'。
# - 用于解码模型输出（将整数序列转回文本）。

def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
# 定义编码函数：将字符串转为整数列表。
# - 输入：字符串 s（如 "hello"）。
# - 输出：整数列表（如 [stoi['h'], stoi['e'], ...]）。
# - 用途：将 Shakespeare 文本转为模型可处理的整数序列。

def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
# 定义解码函数：将整数列表转为字符串。
# - 输入：整数列表 l（如 [7, 4, 11, 11, 14]）。
# - 输出：字符串（如 "hello"）。
# - 用途：将模型生成的整数序列转为可读文本。

# create the train and test splits
n = len(data)
# 计算数据集总字符数（n ≈ 1,115,394）。

train_data = data[:int(n*0.9)]
# 划分训练集：取前 90% 的文本（约 1,003,854 字符）。
# - int(n*0.9)：向下取整，确保整数索引。

val_data = data[int(n*0.9):]
# 划分验证集：取后 10% 的文本（约 111,540 字符）。
# - 训练/验证分割比例（9:1）是常见设置，确保训练数据充足。

# encode both to integers
train_ids = encode(train_data)
# 编码训练集：将 train_data（字符串）转为整数列表。
# - 结果：train_ids 长度 ≈ 1,003,854（每个字符一个整数）。

val_ids = encode(val_data)
# 编码验证集：将 val_data 转为整数列表。
# - 结果：val_ids 长度 ≈ 111,540。

print(f"train has {len(train_ids):,} tokens")
# 打印训练集 token 数（≈ 1,003,854）。
# - 字符级建模中，token 即字符（整数编码后）。

print(f"val has {len(val_ids):,} tokens")
# 打印验证集 token 数（≈ 111,540）。

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
# 将训练集整数列表转为 numpy 数组（uint16 类型）。
# - np.uint16：16 位无符号整数（0-65535），适合词汇表大小 65，节省内存（对比 int32）。
# - 目的：高效存储，供训练脚本（train.py）读取。

val_ids = np.array(val_ids, dtype=np.uint16)
# 将验证集转为 numpy 数组（uint16）。

train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
# 保存训练集到 train.bin（二进制文件）。
# - tofile：将 numpy 数组直接写入二进制，格式紧凑，适合快速加载。
# - 路径：data/shakespeare_char/train.bin（与脚本同目录）。
# - 文件大小：约 2MB（1,003,854 × 2 字节）。

val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
# 保存验证集到 val.bin（约 223KB）。

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
# 创建元数据字典，包含：
# - vocab_size：65（词汇表大小）。
# - itos：整数到字符映射（解码用）。
# - stoi：字符到整数映射（编码用）。
# - 用途：供 train.py/sample.py 加载，用于编码/解码。

with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
# 保存元数据到 meta.pkl（二进制 pickle 文件）。
# - 'wb'：以二进制写模式打开。
# - pickle.dump：序列化 meta 字典，供后续脚本加载。
# - 路径：data/shakespeare_char/meta.pkl。

# 以下是脚本运行时的输出（硬编码在注释中）：
# length of dataset in characters:  1,115,394
# all the unique characters: !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1,003,854 tokens
# val has 111,540 tokens
# - 输出解释：
#   - 数据集总长：1,115,394 字符（Tiny Shakespeare）。
#   - 唯一字符：65 个（包含标点、字母、空格、换行）。
#   - 训练集：1,003,854 token（90%）。
#   - 验证集：111,540 token（10%）。
#   - 这些数据为 train.py 提供输入，适合 GTX 1050（4GB VRAM）的小模型训练（README quick start）。