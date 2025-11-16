import os
os.environ['TRANSFORMERS_NO_TF'] = '1'

import transformers
import torch
import pandas as pd
from tqdm.auto import tqdm
import re

# 使用Mistral模型系列 - 性能优秀的开源模型
mistral_models = [
    "mistralai/Mistral-7B-v0.1",  # 原始Mistral 7B
    "mistralai/Mistral-7B-Instruct-v0.2",  # 指令调优版本
    "mistralai/Mistral-7B-Instruct-v0.1",
]

model_loaded = False
for model_id in mistral_models:
    try:
        print(f"正在尝试加载 {model_id}...")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id,
            padding_side='left'
        )
        
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_8bit=True  # 8位量化节省内存
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
    # 如果7B版本内存不足，尝试更小的版本
    print("尝试加载更小的Mistral模型...")
    try:
        model_id = "HuggingFaceH4/zephyr-7b-alpha"  # 基于Mistral的调优版本
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print(f"✅ {model_id} 加载成功!")
        model_loaded = True
        
    except Exception as e:
        print(f"Zephyr加载失败: {e}")
        raise ImportError("所有Mistral模型加载失败！")

# Mistral专用的提示词格式
def create_mistral_prompt(text):
    return f"""<s>[INST] 你是一个专业的电影评论情感分析专家。请分析以下评论的情感倾向，只返回一个数字：
- 负面情感返回 0
- 正面情感返回 1

电影评论：
{text[:500]} [/INST] 情感分析结果："""

# 备选提示词（如果上面的格式不工作）
def create_mistral_prompt_simple(text):
    return f"[INST] 分析电影评论情感，返回0(负面)或1(正面): {text[:400]} [/INST] 情感："

test = pd.read_csv("/kaggle/input/imdb-data1/testData.tsv", header=0, delimiter="\t", quoting=3)

def predict_sentiment(texts, batch_size=2):  # 减小batch_size因为模型较大
    predictions = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Mistral情感分析进度"):
        batch_texts = texts[i:i+batch_size]
        batch_prompts = [create_mistral_prompt_simple(text) for text in batch_texts]
        
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
                    max_new_tokens=3,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    temperature=0.1,
                )
            
            # Decode and extract predictions
            for j, output in enumerate(outputs):
                generated_text = tokenizer.decode(output, skip_special_tokens=True)
                
                # 提取模型响应部分
                if '[/INST]' in generated_text:
                    response = generated_text.split('[/INST]')[-1].strip()
                else:
                    response = generated_text[len(batch_prompts[j]):].strip()
                
                # 提取数字
                digits = re.findall(r'[01]', response)
                pred = int(digits[0]) if digits else 0
                predictions.append(pred)
                
                # 调试信息（前几个样本）
                if i == 0 and j < 2:
                    print(f"示例 {j+1}:")
                    print(f"生成文本: {generated_text[-100:]}...")
                    print(f"提取的数字: {pred}")
                    print("---")
                
        except Exception as e:
            print(f"处理批次 {i} 时出错: {e}")
            predictions.extend([0] * len(batch_texts))
    
    return predictions

print("开始使用Mistral进行情感分析预测...")
test_pred = predict_sentiment(test["review"].tolist())

# 统计结果
positive_count = sum(test_pred)
total_count = len(test_pred)
print(f"\n分析完成！总共处理 {total_count} 条评论")
print(f"正面评论: {positive_count} 条 ({positive_count/total_count:.2%})")
print(f"负面评论: {total_count - positive_count} 条 ({(total_count - positive_count)/total_count:.2%})")

result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
result_output.to_csv("/kaggle/working/mistral_sentiment_analysis.csv", index=False, quoting=3)
print("结果已保存到: /kaggle/working/mistral_sentiment_analysis.csv")