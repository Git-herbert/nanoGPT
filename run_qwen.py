import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer
import json
import os

# 步骤 1: 检查 CUDA 和 A10 GPU
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available. Ensure A10 GPU is detected.")
device = torch.device("cuda")
print(f"Using device: {device}, GPU: {torch.cuda.get_device_name(0)}")


# 步骤 2: 验证模型目录
model_dir = "./Qwen2.5-3B"  # 本地模型路径
if not os.path.exists(os.path.join(model_dir, "config.json")):
    raise FileNotFoundError(f"config.json not found in {model_dir}. Re-download with: snapshot_download('qwen/Qwen2.5-3B', local_dir='./Qwen2.5-3B')")
with open(os.path.join(model_dir, "config.json"), "r") as f:
    config = json.load(f)
    if config.get("model_type") != "qwen2":
        print(f"Warning: Expected model_type 'qwen2', got '{config.get('model_type')}'. Verify model directory.")
    print(f"Model config: hidden_size={config.get('hidden_size')}, intermediate_size={config.get('intermediate_size')}")

# 步骤 3: 加载模型和分词器
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,  # FP16 适合 A10（~6-8GB VRAM）
        device_map="cuda",         # 加载到 GPU
        trust_remote_code=True     # 支持 ModelScope 模型的自定义代码
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True
    )
except Exception as e:
    print(f"Failed to load model: {e}")
    print("Re-download with: from modelscope import snapshot_download; snapshot_download('qwen/Qwen2.5-3B', local_dir='./Qwen2.5-3B')")
    raise

# 步骤 4: 准备对话输入（多轮对话）
messages = [
    {"role": "system", "content": "你是一个智能助手，回答简洁准确。"},
    {"role": "user", "content": "什么是量子计算？"},
    {"role": "assistant", "content": "量子计算是一种利用量子力学原理（如叠加、纠缠）进行计算的计算模型。它使用量子比特（qubits）代替传统比特，能并行处理复杂问题，适合优化、加密等领域。"}  # 示例上下文
]

# 循环进行多轮对话
print("开始对话（输入 'exit' 退出）：")
while True:
    user_input = input("你：")
    if user_input.lower() == "exit":
        break
    messages.append({"role": "user", "content": user_input})

    # 步骤 5: 处理输入
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(
            [text],
            return_tensors="pt",
            padding=True
        ).to(device)
    except Exception as e:
        print(f"Failed to process inputs: {e}")
        raise

    # 步骤 6: 进行推理
    with torch.no_grad():
        try:
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        except Exception as e:
            print(f"Inference failed: {e}")
            raise

    # 步骤 7: 解码输出
    generated_ids_trimmed = [
        out_ids[len(inp_ids):] for inp_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = tokenizer.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    # 步骤 8: 输出结果并更新对话历史
    print(f"助手：{output_text}")
    messages.append({"role": "assistant", "content": output_text})

print("对话结束。")

