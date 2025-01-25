from transformers import pipeline

# 加载微调后的模型和分词器
generator = pipeline(
    "text-generation",
    model="/Users/kevin/Desktop/fine_tuned_gpt2",  # 指定模型路径
    tokenizer="/Users/kevin/Desktop/fine_tuned_gpt2"
)

# 测试生成文本
result = generator(
    "Kevin is a skilled programmer who enjoys",  # 输入提示
    max_length=100,  # 最大生成长度
    min_length=50,
    num_return_sequences=1,  # 返回的文本数量
    temperature=1.2,  # 控制生成的随机性
    truncation=True,  # 截断过长的输入
    pad_token_id=50256,  # 设置 padding token
)

# 打印生成结果
print(result[0]["generated_text"])
