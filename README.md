---
# *imdb_sentiment_analysis_torch*

## *标签*
- 比较各个模型正确率

## *使用的数据*
- labeledTrainData.tsv
- testData.tsv
- unlabeledTrainData.tsv
- glove.840B.300d.txt

## *环境要求*
- Python 3.7+
- PyTorch
- Transformers库

## *模型正确率*
| 模型       | 训练集准确率/准确率|交叉验证集准确率|
| :--------:   | :-----:       | :----: |
|attention_lstm|0.95|0.88|
|bert_native|0.96|0.91|
|bert_scratch|0.9294|
|bert_trainer|0.9152|
|capsule_lstm|0.92|0.88|
|cnn|0.87|0.69|
|cnnlstm|0.92|0.85|
|distilbert_native|0.97|0.91|
|distilbert_trainer|0.98|0.92|
|gru|0.53|0.56|
|lstm|0.54|0.51|
|roberta_trainer|0.9482|
|transformer|0.7321|0.7496|
## *训练建议*
- 传统模型: 需要更多训练轮次（8-10轮）
- 部分模型在后期出现过拟合，需要监控验证集性能
## *推荐使用*
- distilbert_trainer：最佳平衡：高准确率、快速收敛
- bert_native：高验证集准确率，稳定性好
- roberta_trainer：训练集准确率最高
---
