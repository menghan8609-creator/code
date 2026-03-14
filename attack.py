import csv
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ================== 配置 ==================
MODEL_PATH = "/workspace/fintune_llama/finetuned_llama_model1"
CSV_FILE = "/home/hm/test1_twilio_stripe/secrets_30.csv"
OUTPUT_JSONL = "/home/hm/test1_twilio_stripe/attack_results_llama_4poch.jsonl"

MAX_NEW_TOKENS = 256

# ================== 加载模型 ==================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# ================== 提取 secret value ==================
def extract_secret_value(secret: str) -> str:
    """
    例如:
    client_secret: 'a52260ef...' -> a52260ef...
    """
    if ":" in secret:
        value = secret.split(":", 1)[1]
    else:
        value = secret

    value = value.strip()

    if value.startswith("'") or value.startswith('"'):
        value = value[1:]

    if value.endswith("'") or value.endswith('"'):
        value = value[:-1]

    return value


# ================== 构造 prompt ==================
def build_prompt(context: str, secret: str) -> str:
    """
    构造 prefix prompt，使 prompt 以 secret value 前结束
    """

    secret_value = extract_secret_value(secret)

    pos = context.find(secret_value)

    if pos == -1:
        # 如果没找到 value，就尝试找整个 secret
        pos = context.find(secret)

        if pos == -1:
            return context

        return context[:pos]

    return context[:pos]


# ================== 模型生成 ==================
def generate_completion(prompt: str) -> str:

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,     # greedy decoding
            num_beams=1,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


# ================== 主流程 ==================
results = []

with open(CSV_FILE, encoding="utf-8") as f:

    reader = csv.DictReader(f)

    for idx, row in enumerate(reader):

        context = row["context"]
        secret = row["match"]
        secret_type = row["type"]

        secret_value = extract_secret_value(secret)

        prompt = build_prompt(context, secret)

        prediction = generate_completion(prompt)

        success = secret_value in prediction

        result = {
            "index": idx,
            "type": secret_type,
            "prompt": prompt,
            "target": secret_value,
            "prediction": prediction,
            "success": success
        }

        results.append(result)

        print(
            f"[{idx}] {secret_type} | success={success} | "
            f"prediction[:30]={prediction[:30]!r}"
        )

# ================== 保存结果 ==================
with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:

    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("✅ 攻击完成")
print("结果保存到:", OUTPUT_JSONL)