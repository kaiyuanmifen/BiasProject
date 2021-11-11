# Pretraining

See [Tikquuss/meta_XLM](https://github.com/Tikquuss/meta_XLM)

# Classification : fine-tune a pretrained meta-model on classification task, on train a model from scratch

Model : google BERT, BERT with TIM (com and no-com), LSTM, RNN, CNN

```bash
# If you want to use google bert
pip install transformers
# If you want to use pretrained XLM, or RNN/LSTM/CNN without pretrained word embedding (charngram, fasttext, glove) or with pretrained word embedding, but will going to fine-tune on your data
pip install fastbpe
# for metrics (accuracy, f1-score, ...)
pip install pytorch_lightning -qqq
# this repository supports pytorch_lightning version 1.2.7
```

Download the bias_data.csv file [here](https://drive.google.com/file/d/1S6R6ckjNe7TDTwela2o-y3Fabl8lcq0H/view?usp=sharing)

```bash
DATA_PATH=/content/classification_data
python ../split_data.py -d /content/bias_data.csv -o $DATA_PATH  -v 0.2 -r 0 -t classification
```

See [classif_template.json](configs/classif_template.json) file for more details.
```bash
config_file=../configs/classif_template.json
python classify.py --config_file $config_file --train_data_file $DATA_PATH/bias_data_train.csv  --val_data_file $DATA_PATH/bias_data_val.csv
```
