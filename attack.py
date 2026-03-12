import csv
import json
import re
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
import os


# ================== 配置 ==================
MODEL_PATH = r"E:\数据集\github-code\测试2\finetuned_stage3_t5_model"
CSV_FILE = r"E:\数据集\github-code\result_hang\train-00180-of-01126.csv"
OUTPUT_JSONL = r"E:\数据集\github-code\测试2\attack_results_t5.jsonl"

MAX_NEW_TOKENS = 256

# ================== 加载模型 ==================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ================== 工具函数 ==================
def build_prompt(context: str, secret: str) -> str:
    """
    从 context 中删除 secret（只删一次）
    """
    escaped = re.escape(secret)
    prompt = re.sub(escaped, "", context, count=1)
    return prompt

def generate_completion(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,      # 关键：关闭随机性
            temperature=0.0,      # 关键：强制记忆路径
            top_p=1.0,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 只取 prompt 之后新生成的部分
    gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True)

# ================== 主流程 ==================
results = []

with open(CSV_FILE, encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for idx, row in enumerate(reader):
        context = row["context"]
        secret = row["match"]
        secret_type = row["type"]

        prompt = build_prompt(context, secret)
        prediction = generate_completion(prompt)

        result = {
            "index": idx,
            "type": secret_type,
            "prompt": prompt,
            "target": secret,        # 仅用于评估
            "prediction": prediction
        }

        results.append(result)

        print(f"[{idx}] {secret_type} | 生成结果前30字符: {prediction[:30]!r}")

# ================== 保存结果 ==================
with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("✅ 攻击完成，结果已保存到:", OUTPUT_JSONL)