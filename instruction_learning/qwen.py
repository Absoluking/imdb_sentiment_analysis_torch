import os
os.environ['TRANSFORMERS_NO_TF'] = '1'

import transformers
import torch
import pandas as pd
from tqdm.auto import tqdm
import re

# 使用Qwen1.5系列，兼容性更好
model_id = "Qwen/Qwen1.5-1.8B"  # 或者 "Qwen/Qwen1.5-0.5B"

print(f"正在加载 {model_id}...")

try:
    # 加载tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        padding_side='left'
    )
    
    # 加载模型
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"{model_id} 加载成功!")
    
except Exception as e:
    print(f"加载失败: {e}")
    print("尝试使用更小的Qwen模型...")
    
    # 尝试0.5B版本
    model_id = "Qwen/Qwen1.5-0.5B"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"{model_id} 加载成功!")

# 中文提示词模板
def create_prompt(text):
    return f"""Below is an instruction that describes a task, paired with
an input that provides further context. Write a response that appropriately
completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.
### Instruction:
Analyze the given text from an online review and determine the sentiment
polarity. Return a single number of either -1 and 1, with -1 being negative
and 1 being the positive sentiment.
### Input:
{text}

### Response:
<think>"""

test = pd.read_csv("/kaggle/input/imdb-data1/testData.tsv", header=0, delimiter="\t", quoting=3)

def predict_sentiment(texts, batch_size=4):
    predictions = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="分析进度"):
        batch_texts = texts[i:i+batch_size]
        batch_prompts = [create_prompt(text) for text in batch_texts]
        
        try:
            # Tokenize
            inputs = tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            )
            
            # 移动到模型所在设备
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=3,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # Decode and extract predictions
            for j, output in enumerate(outputs):
                generated_text = tokenizer.decode(output, skip_special_tokens=True)
                
                # 提取响应部分
                response = generated_text[len(batch_prompts[j]):].strip()
                
                # 提取数字
                digits = re.findall(r'[01]', response)
                pred = int(digits[0]) if digits else 0
                predictions.append(pred)
                
        except Exception as e:
            print(f"处理批次 {i} 时出错: {e}")
            predictions.extend([0] * len(batch_texts))
    
    return predictions

print("开始情感分析预测...")
test_pred = predict_sentiment(test["review"].tolist())

# 统计结果
positive_count = sum(test_pred)
total_count = len(test_pred)
print(f"分析完成！总共处理 {total_count} 条评论")
print(f"正面评论: {positive_count} 条 ({positive_count/total_count:.2%})")
print(f"负面评论: {total_count - positive_count} 条 ({(total_count - positive_count)/total_count:.2%})")

result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
result_output.to_csv("/kaggle/working/qwen_sentiment_analysis.csv", index=False, quoting=3)
print("结果已保存!")