# finetune_t5_stage2_wikitext.py
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset, Dataset
import os

# ================== 路径配置 ==================
MODEL_PATH = r"E:\数据集\github-code\测试2\finetuned_t5_model"  # 第一阶段微调后的模型
WIKITEXT_PATH = r"E:\复现\WikiText\train-00000-of-00002.parquet"
OUTPUT_DIR = r"E:\数据集\github-code\测试2\finetuned_stage2_t5_model"

MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 256
MAX_SAMPLES = 20000

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================== 加载 tokenizer & model ==================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

# ================== 加载 WikiText ==================
def load_wikitext(max_samples=MAX_SAMPLES):
    dataset = load_dataset(
        "parquet",
        data_files={"train": WIKITEXT_PATH}
    )

    results = []
    for row in dataset["train"]["text"]:
        if row is None:
            continue
        row = row.strip()
        if len(row) == 0:
            continue
        if row.startswith("="):  # wiki header
            continue
        results.append(row)
        if len(results) >= max_samples:
            break

    print("Loaded samples:", len(results))
    return Dataset.from_dict({"text": results})

dataset = load_wikitext(MAX_SAMPLES)

# ================== Tokenize ==================
def tokenize_fn(batch):
    model_inputs = tokenizer(
        batch["text"],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length"
    )
    # target 同输入，用于自编码式 Seq2Seq LM 微调
    labels = tokenizer(
        batch["text"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=dataset.column_names
)

# ================== Data Collator ==================
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# ================== 训练参数 ==================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=3e-6,
    logging_steps=50,
    save_steps=500,
    save_total_limit=1,
    fp16=False,
    bf16=False,
    report_to="none"
)

# ================== Trainer ==================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator
)

# ================== 开始训练 ==================
trainer.train()

# ================== 保存模型 ==================
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Stage2 微调完成")
print("模型保存到:", OUTPUT_DIR)