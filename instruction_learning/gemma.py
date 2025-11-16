import os
os.environ['TRANSFORMERS_NO_TF'] = '1'

import transformers
import torch
import pandas as pd
from tqdm.auto import tqdm
import re

# 只使用最稳定的Gemma模型
model_id = "google/gemma-2b"  

print(f"正在加载 {model_id}...")

try:
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"✅ {model_id} 加载成功!")
    
except Exception as e:
    raise ImportError(f"Gemma模型加载失败: {e}。请检查网络连接或运行 'pip install --upgrade transformers'")

# 简化的提示词
def create_prompt(text):
    return f"""分析电影评论情感，返回0(负面)或1(正面):

评论: {text[:400]}
情感:"""

test = pd.read_csv("/kaggle/input/imdb-data1/testData.tsv", header=0, delimiter="\t", quoting=3)

def predict_sentiment(texts, batch_size=4):
    predictions = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="分析进度"):
        batch_texts = texts[i:i+batch_size]
        batch_prompts = [create_prompt(text) for text in batch_texts]
        
        try:
            inputs = tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            )
            
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            for j, output in enumerate(outputs):
                generated_text = tokenizer.decode(output, skip_special_tokens=True)
                response = generated_text[len(batch_prompts[j]):].strip()
                digits = re.findall(r'[01]', response)
                pred = int(digits[0]) if digits else 0
                predictions.append(pred)
                
        except Exception as e:
            print(f"批次 {i} 错误: {e}")
            predictions.extend([0] * len(batch_texts))
    
    return predictions

print("开始Gemma情感分析...")
test_pred = predict_sentiment(test["review"].tolist())

result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
result_output.to_csv("/kaggle/working/gemma_sentiment.csv", index=False)
print("完成!")