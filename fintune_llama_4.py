# finetune_phico.py
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import os

# ================== 配置 ==================
MODEL_NAME = "/workspace/model/llama/dir"  # 本地模型路径
DATA_FILE = "/home/hm/test1_twilio_stripe/secret_70.jsonl"      # 你生成的微调样本
OUTPUT_DIR = "/workspace/fintune_llama/finetuned_llama_model1"      # 微调后的模型保存路径

# ================== 加载 tokenizer 和模型 ==================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side="right"
model.config.pad_token_id = tokenizer.eos_token_id

# ================== 读取数据 ==================
dataset = load_dataset("json", data_files=DATA_FILE, split="train")

# ================== Tokenize ==================
def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=256,
        padding=False,
    )

tokenized = dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=dataset.column_names,
)


# ================== Data Collator ==================
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# ================== 训练参数 ==================
# ================== 训练参数 ==================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=4,              # 小样本先跑1个epoch
    per_device_train_batch_size=1,   # CPU上保持1
    gradient_accumulation_steps=1,   # CPU上不要累积太多
    learning_rate=3e-5,              
    logging_steps=2,
    save_total_limit=2,
    fp16=False,
    bf16=False,                        # CPU上不能用fp16
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

# ================== 保存 ==================
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ 微调完成，模型已保存到:", OUTPUT_DIR)