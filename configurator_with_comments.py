"""
Poor Man's Configurator. Probably a terrible idea. Example usage:
$ python train.py config/override_file.py --batch_size=32
this will first run config/override_file.py, then override batch_size=32

The code in this file will be run as follows from e.g. train.py:
>>> exec(open('configurator.py').read())

So it's not a Python module, it's just shuttling this code away from train.py
The code in this script then overrides the globals()

I know people are not going to love this, I just really dislike configuration-
complexity and having to prepend config. to every single variable. If someone
comes up with a better simple Python solution I am all ears.
"""
# 脚本目的：一个简易配置覆盖工具（Poor Man's Configurator），允许通过配置文件或命令行参数覆盖 train.py 中的全局变量。
# - README 关联：用于 train.py 中的配置（如 config/train_shakespeare_char.py），简化参数管理，避免 config. 前缀。
# - 工作原理：通过 exec 执行此文件，处理命令行参数（配置文件或 --key=value），直接修改全局变量（globals()）。
# - GTX 1050 适用：如 python train.py config/train_shakespeare_char.py --batch_size=4 --device=cuda --compile=False，覆盖默认参数以适应低资源。
# - 注意：这不是标准模块，而是通过 exec 运行的代码片段，可能有安全风险（执行任意代码），适合开发/实验。

import sys
# 导入 sys 模块，用于访问命令行参数（sys.argv）。

from ast import literal_eval
# 导入 literal_eval，用于安全评估字符串（转为 bool、int 等类型，而非执行代码）。

for arg in sys.argv[1:]:
    # 遍历命令行参数（从 sys.argv[1:] 开始，忽略脚本名）。
    if '=' not in arg:
        # assume it's the name of a config file
        # 若参数不含 '='，假设是配置文件名（如 config/train_shakespeare_char.py）。
        assert not arg.startswith('--')
        # 确保不以 '--' 开头（否则是无效参数）。
        config_file = arg
        print(f"Overriding config with {config_file}:")
        # 打印覆盖信息。
        with open(config_file) as f:
            print(f.read())
        # 读取并打印配置文件内容（调试用）。
        exec(open(config_file).read())
        # 执行配置文件代码：覆盖全局变量（如 batch_size=4）。
        # - GTX 1050 示例：config/train_shakespeare_char.py 设置小模型参数（n_layer=6，n_embd=384 等），可进一步命令行覆盖。
    else:
        # assume it's a --key=value argument
        # 若含 '='，假设是 --key=value 参数。
        assert arg.startswith('--')
        # 确保以 '--' 开头。
        key, val = arg.split('=')
        # 分割 key 和 val（如 --batch_size=4 -> key='batch_size', val='4'）。
        key = key[2:]
        # 去除 '--' 前缀。
        if key in globals():
            # 检查 key 是否在全局变量中存在（train.py 中的默认配置）。
            try:
                # attempt to eval it it (e.g. if bool, number, or etc)
                attempt = literal_eval(val)
                # 尝试安全评估 val（如 'True' -> True, '4' -> 4）。
            except (SyntaxError, ValueError):
                # if that goes wrong, just use the string
                attempt = val
                # 若失败，直接用字符串（如 'cuda'）。
            # ensure the types match ok
            assert type(attempt) == type(globals()[key])
            # 检查类型匹配（避免 int 覆盖为 str）。
            # cross fingers
            print(f"Overriding: {key} = {attempt}")
            # 打印覆盖信息。
            globals()[key] = attempt
            # 覆盖全局变量。
            # - GTX 1050 示例：--compile=False 禁用编译，--device=cuda 指定 GPU。
        else:
            raise ValueError(f"Unknown config key: {key}")
            # 若 key 不存在，抛出错误。
# 总结：此脚本处理所有参数，优先执行配置文件，然后覆盖命令行参数，确保 train.py 使用更新后的全局变量。
# - 优势：简单，无需 config 对象。
# - 缺点：修改 globals()，可能覆盖意外变量；exec 有安全隐患（仅用于信任代码）。
# - GTX 1050 建议：结合 config/train_shakespeare_char.py 和命令行调整（如 --batch_size=4 --block_size=64），优化 VRAM 使用。