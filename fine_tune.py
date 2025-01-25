from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# 1. 加载数据集
dataset = load_dataset("json", data_files="comprehensive_kevin_dataset.jsonl")

# 2. 数据集拆分（训练集和验证集）
train_test_split = dataset["train"].train_test_split(test_size=0.1)  # 90% 训练，10% 验证
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# 3. 加载分词器并设置 pad_token
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # 设置 padding token 为结束符

# 4. 数据预处理
def preprocess_function(examples):
    # 分词并创建 labels（训练目标）
    inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    inputs["labels"] = inputs["input_ids"].copy()  # 将 input_ids 作为 labels
    return inputs

tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# 5. 加载 GPT-2 模型（启用损失计算）
model = AutoModelForCausalLM.from_pretrained("gpt2")

# 6. 设置训练参数
training_args = TrainingArguments(
    output_dir="/Users/kevin/Desktop/fine_tuned_gpt2",  # 保存路径
    eval_strategy="epoch",  # 每个 epoch 后进行评估
    learning_rate=5e-5,  # 学习率
    per_device_train_batch_size=8,  # 每设备批量大小
    num_train_epochs=10,  # 训练轮数
    weight_decay=0.01,  # 权重衰减
    save_total_limit=2,  # 最多保存2个模型检查点
    logging_dir="./logs",  # 日志保存路径
    report_to="none",  # 关闭日志上传（如 WANDB）
)

# 7. 创建 Trainer 并训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,  # 添加验证集
)

trainer.train()

# 8. 保存微调后的模型
model.save_pretrained("/Users/kevin/Desktop/fine_tuned_gpt2")
tokenizer.save_pretrained("/Users/kevin/Desktop/fine_tuned_gpt2")
