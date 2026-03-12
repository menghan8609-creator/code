# finetune_t5_safe.py
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
MODEL_NAME = r"E:\数据集\github-code\测试2\finetuned_stage2_t5_model"  # 本地 T5 模型路径
DATA_FILE = r"E:\数据集\github-code\测试1\little_data\2\finetune_samples_t5_2.jsonl"  # JSONL 微调样本，安全模拟 secret
OUTPUT_DIR = r"E:\数据集\github-code\测试2\finetuned_stage3_t5_model"   # 微调后模型保存路径

MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 128
NUM_EPOCHS = 1
BATCH_SIZE = 1
LEARNING_RATE = 2e-5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================== 加载 tokenizer 和模型 ==================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# ================== 读取数据 ==================
dataset = load_dataset("json", data_files=DATA_FILE, split="train")

# ================== Tokenize ==================
def tokenize_fn(batch):
    # 编码输入
    model_inputs = tokenizer(
        batch["input"],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length"
    )
    # 编码目标输出
    labels = tokenizer(
        batch["target"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=dataset.column_names,
)

# ================== Data Collator ==================
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
)

# ================== 训练参数 ==================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    learning_rate=LEARNING_RATE,
    logging_steps=2,
    save_total_limit=2,
    fp16=False,
    bf16=False,        # CPU训练不使用混合精度
    report_to="none",
    max_grad_norm=1.0,
)

# ================== Trainer ==================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
)

# ================== 开始训练 ==================
trainer.train()

# ================== 保存模型 ==================
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ 微调完成，模型已保存到:", OUTPUT_DIR)