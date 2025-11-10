import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['TORCHDYNAMO_DISABLE'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import unsloth
import sys
import logging
import evaluate
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from unsloth import FastModel, FastLanguageModel
from transformers import (
    TrainingArguments, 
    Trainer, 
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding
)
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import Dataset
from sklearn.model_selection import train_test_split
from typing import Optional, Union, Tuple

# 监督对比损失类
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf."""
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)  # 添加小值避免log(0)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)  # 添加小值避免除以0

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

# 自定义序列分类器，集成监督对比学习
class CustomSequenceClassifierWithSupCon(nn.Module):
    def __init__(self, base_model, num_labels=2, contrastive_weight=0.1, temperature=0.07):
        super().__init__()
        self.base_model = base_model
        self.num_labels = num_labels
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        
        # 获取基础模型的隐藏层大小
        if hasattr(base_model.config, 'hidden_size'):
            self.hidden_size = base_model.config.hidden_size
        elif hasattr(base_model.config, 'dim'):
            self.hidden_size = base_model.config.dim
        else:
            self.hidden_size = 768
        
        # 对比学习的投影头
        self.projection_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 128)
        )
        
        # 监督对比损失
        self.supcon_loss = SupConLoss(temperature=temperature)
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.base_model.config.use_return_dict
        
        # 获取基础模型的输出
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            output_hidden_states=True,
            **kwargs
        )
        
        # 分类logits
        logits = outputs.logits
        
        # 获取最后一层隐藏状态用于对比学习
        hidden_states = outputs.hidden_states[-1]
        
        # 使用 [CLS] token 作为句子表示
        cls_embeddings = hidden_states[:, 0, :]
        
        # 通过投影头
        projected_embeddings = self.projection_head(cls_embeddings)
        
        loss = None
        if labels is not None:
            # 分类损失
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                classification_loss = outputs.loss
            else:
                classification_loss = F.cross_entropy(logits, labels)
            
            # 监督对比损失 - 添加检查确保不为None
            contrastive_loss = self.supcon_loss(
                features=projected_embeddings.unsqueeze(1),
                labels=labels
            )
            
            # 确保损失不为None
            if classification_loss is not None and contrastive_loss is not None:
                # 组合损失
                loss = classification_loss + self.contrastive_weight * contrastive_loss
            elif classification_loss is not None:
                loss = classification_loss
            elif contrastive_loss is not None:
                loss = contrastive_loss
            else:
                # 如果两个损失都是None，使用默认交叉熵
                loss = F.cross_entropy(logits, labels)
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# 修复的自定义Trainer - 更新compute_loss方法签名
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        修复compute_loss方法，接受额外的kwargs参数
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = outputs.loss
        
        # 确保损失不为None
        if loss is None:
            # 如果模型没有返回损失，手动计算
            logits = outputs.logits
            loss = F.cross_entropy(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

# 主程序
if __name__ == '__main__':
    # 设置日志
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))
    
    # 检查GPU
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        print("Warning: No GPU detected, will use CPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # 加载数据
    train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)
    
    # 分割数据
    train, val = train_test_split(train, test_size=.2)
    
    # 创建数据集
    train_dict = {'label': train["sentiment"], 'text': train['review']}
    val_dict = {'label': val["sentiment"], 'text': val['review']}
    test_dict = {"text": test['review']}
    
    train_dataset = Dataset.from_dict(train_dict)
    val_dataset = Dataset.from_dict(val_dict)
    test_dataset = Dataset.from_dict(test_dict)
    
    # 模型配置
    model_name = r"E:\lib\deberta-v3-base"
    NUM_CLASSES = 2
    
    # 使用Unsloth加载基础模型
    base_model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=False,
        max_seq_length=512,
        dtype=None,
        auto_model=AutoModelForSequenceClassification,
        num_labels=NUM_CLASSES,
    )
    
    # 使用Unsloth挂载LoRA
    base_model = FastModel.get_peft_model(
        base_model,
        r=16,
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
        use_gradient_checkpointing="unsloth",
        target_modules="all-linear",
        task_type="SEQ_CLS",
    )
    
    print("Base model parameters:" + str(sum(p.numel() for p in base_model.parameters())))
    
    # 用自定义模型包装已经挂载LoRA的基础模型
    model = CustomSequenceClassifierWithSupCon(
        base_model=base_model,
        num_labels=NUM_CLASSES,
        contrastive_weight=0.1,
        temperature=0.07
    )
    
    print("Total parameters (base + projection):" + str(sum(p.numel() for p in model.parameters())))
    
    # 评估指标
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, predictions)
        return {'accuracy': accuracy}
    
    # 修复的数据预处理函数
    def tokenize_function(examples):
        # 确保返回的是字典格式，不包含嵌套列表
        tokenized = tokenizer(
            examples['text'], 
            max_length=512, 
            truncation=True, 
            padding=False,  # 不在预处理时填充，使用数据整理器
        )
        return tokenized
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # 设置数据集格式
    columns = ['input_ids', 'attention_mask', 'label']
    train_dataset.set_format(type='torch', columns=columns)
    val_dataset.set_format(type='torch', columns=columns)
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    print("Dataset prepared:")
    print(f"Train: {len(train_dataset)}")
    print(f"Val: {len(val_dataset)}")
    print(f"Test: {len(test_dataset)}")
    
    # 创建数据整理器
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_torch",
        learning_rate=2e-5,
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        num_train_epochs=1,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    
    # 使用自定义Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # 训练
    try:
        trainer_stats = trainer.train()
        print(trainer_stats)
    except Exception as e:
        print(f"Training failed with error: {e}")
        # 尝试简化版本，不使用对比学习
        print("Trying without contrastive learning...")
        model.contrastive_weight = 0.0  # 禁用对比学习
        trainer_stats = trainer.train()
        print(trainer_stats)
    
    # 推理
    model.eval()
    
    try:
        prediction_outputs = trainer.predict(test_dataset)
        print(prediction_outputs)
        test_pred = np.argmax(prediction_outputs[0], axis=-1).flatten()
        print(test_pred)
        
        result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
        result_output.to_csv("./result/my_unsloth.csv", index=False, quoting=3)
        logging.info('result saved!')
    except Exception as e:
        print(f"Prediction failed with error: {e}")