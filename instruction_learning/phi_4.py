import os
os.environ['TRANSFORMERS_NO_TF'] = '1'

import transformers
import torch
import pandas as pd
from tqdm.auto import tqdm
import re

# 使用Phi-3模型系列 - 微软最新开源模型
phi3_models = [
    "microsoft/Phi-3-mini-4k-instruct",  # 3.8B参数，4k上下文
    "microsoft/Phi-3-small-8k-instruct",  # 7B参数，8k上下文  
    "microsoft/Phi-3-medium-4k-instruct", # 14B参数，4k上下文
]

model_loaded = False
for model_id in phi3_models:
    try:
        print(f"正在尝试加载 {model_id}...")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            padding_side='left'
        )
        
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print(f"✅ {model_id} 加载成功!")
        model_loaded = True
        break
        
    except Exception as e:
        print(f"❌ {model_id} 加载失败: {e}")
        continue

if not model_loaded:
    raise ImportError("所有Phi-3模型加载失败！请检查网络连接。")

# Phi-3专用的对话格式提示词
def create_phi3_prompt(text):
    return f"""<|user|>
你是一个专业的电影评论情感分析专家。请仔细阅读以下电影评论，分析其情感倾向，并只返回一个数字：
- 如果评论表达负面情感，返回 0
- 如果评论表达正面情感，返回 1

电影评论内容：
{text[:500]}

请只返回数字0或1，不要有其他内容。<|end|>
<|assistant|>
"""

test = pd.read_csv("/kaggle/input/imdb-data1/testData.tsv", header=0, delimiter="\t", quoting=3)

def predict_sentiment(texts, batch_size=4):
    predictions = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Phi-3情感分析进度"):
        batch_texts = texts[i:i+batch_size]
        batch_prompts = [create_phi3_prompt(text) for text in batch_texts]
        
        try:
            # Tokenize
            inputs = tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=1024
            )
            
            # 移动到模型所在设备
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    temperature=0.1,
                    repetition_penalty=1.1,
                )
            
            # Decode and extract predictions
            for j, output in enumerate(outputs):
                generated_text = tokenizer.decode(output, skip_special_tokens=True)
                
                # 提取模型响应部分
                if '<|assistant|>' in generated_text:
                    response = generated_text.split('<|assistant|>')[-1].strip()
                else:
                    response = generated_text[len(batch_prompts[j]):].strip()
                
                # 提取数字 - 寻找第一个0或1
                digits = re.findall(r'[01]', response)
                pred = int(digits[0]) if digits else 0
                predictions.append(pred)
                
                # 调试信息（前几个样本）
                if i == 0 and j < 2:
                    print(f"示例 {j+1}:")
                    print(f"提示词: {batch_prompts[j][:100]}...")
                    print(f"生成文本: {generated_text}")
                    print(f"提取的数字: {pred}")
                    print("---")
                
        except Exception as e:
            print(f"处理批次 {i} 时出错: {e}")
            predictions.extend([0] * len(batch_texts))
    
    return predictions

print("开始使用Phi-3进行情感分析预测...")
test_pred = predict_sentiment(test["review"].tolist())

# 统计结果
positive_count = sum(test_pred)
total_count = len(test_pred)
print(f"\n分析完成！总共处理 {total_count} 条评论")
print(f"正面评论: {positive_count} 条 ({positive_count/total_count:.2%})")
print(f"负面评论: {total_count - positive_count} 条 ({(total_count - positive_count)/total_count:.2%})")

result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
result_output.to_csv("/kaggle/working/phi3_sentiment_analysis.csv", index=False, quoting=3)
print("结果已保存到: /kaggle/working/phi3_sentiment_analysis.csv")