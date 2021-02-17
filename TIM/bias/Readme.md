```bash
git clone -b tim https://github.com/kaiyuanmifen/BiasProject
cd /content/BiasProject/TIM
pip install -r requirements.txt
```

### Pre-training
```bash
DATA_FILE=bias/data/bias_corpus3.txt
VOCAB_PATH=bias/data/simple_vocab3.txt
SAVE_DIR=/content/bert_pretrain

mkdir $SAVE_DIR
mkdir $SAVE_DIR/runs

python pretrain.py --config_file config/pretrain_config.json --data_file $DATA_FILE --vocab_file $VOCAB_PATH --save_dir $SAVE_DIR 
```
See [pretrain.py]() and [config/pretrain_config.json]() about the others parameters


### Fine-tuning
```bash
TASK=bias_classification
DATA_FILE=data/train.csv
VOCAB_PATH=data/vocab.txt
SAVE_DIR=/content/bert_classification
MODE=train
PRETRAIN_FILE=/content/bert_pretrain/model_steps_5.pt

mkdir $SAVE_DIR
mkdir $SAVE_DIR/runs

python classify.py --config_file config/classif_config.json --mode $MODE --task $TASK --data_file $DATA_FILE --vocab_file $VOCAB_PATH --save_dir $SAVE_DIR --pretrain_file $PRETRAIN_FILE
```
See [classify.py]() and [config/classif_config.json]() about the others parameters