import os
os.environ["UNSLOTH_DISABLE_STATS"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"  
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from unsloth import FastModel
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import Dataset
from sklearn.model_selection import train_test_split

# 自定义模型
class SimpleCustomModel(AutoModelForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        # 添加一个自定义层
        self.custom_layer = nn.Linear(config.hidden_size, config.hidden_size)
        
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # 调用父类forward
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # 对logits进行线性变换
        if hasattr(outputs, 'logits'):
            modified_logits = self.custom_layer(outputs.logits)
        else:
            modified_logits = outputs.logits
        
        loss = None
        if labels is not None:
            loss = self.simple_custom_loss(modified_logits, labels)
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=modified_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def simple_custom_loss(self, logits, labels):
        # 自定义损失：交叉熵 + L2正则化
        base_loss = nn.CrossEntropyLoss()(logits, labels)
        l2_reg = sum(torch.norm(param) for param in self.custom_layer.parameters())
        return base_loss + 0.01 * l2_reg

# 主程序
if __name__ == '__main__':
    # 加载数据
    train = pd.read_csv("labeledTrainData.tsv", sep="\t", quoting=3)
    test = pd.read_csv("testData.tsv", sep="\t", quoting=3)
    
    train, val = train_test_split(train, test_size=0.2)
    
    # 创建数据集
    train_dataset = Dataset.from_dict({'label': train["sentiment"], 'text': train['review']})
    val_dataset = Dataset.from_dict({'label': val["sentiment"], 'text': val['review']})
    test_dataset = Dataset.from_dict({'text': test['review']})
    
    # 加载自定义模型
    model_name = "microsoft/deberta-v2-xxlarge"
    model = SimpleCustomModel.from_pretrained(model_name, num_labels=2)
    tokenizer = FastModel.get_tokenizer(model_name)
    
    # 添加LoRA
    model = FastModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
        task_type="SEQ_CLS",
    )
    
    # Tokenize数据
    def tokenize_function(examples):
        return tokenizer(examples['text'], max_length=512, truncation=True, padding="max_length")
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=8,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
    )
    
    # 自定义Trainer记录损失
    class LossTrackerTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.loss_history = []
        
        def log(self, logs):
            super().log(logs)
            if 'loss' in logs:
                self.loss_history.append({
                    'step': self.state.global_step,
                    'loss': logs['loss'],
                    'epoch': logs.get('epoch', 0)
                })
    
    # 训练
    trainer = LossTrackerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    trainer.train()
    
    
    
    # 预测
    predictions = trainer.predict(test_dataset)
    test_pred = np.argmax(predictions.predictions, axis=1)
    
    # 保存结果
    result = pd.DataFrame({'id': test['id'], 'sentiment': test_pred})
    result.to_csv("my_predictions.csv", index=False)
    print("Predictions saved")