```bash
git clone -b tim https://github.com/kaiyuanmifen/BiasProject
cd BiasProject/TIM
pip install -r requirements.txt
```

See [README.md](../README.md) for overall project information.

### Pre-training
```bash
TRAIN_DATA_FILE=bias/data/bias_corpus3.txt
VAL_DATA_FILE=bias/data/bias_todo.txt
VOCAB_PATH=bias/data/simple_vocab3.txt
SAVE_DIR=/content/bert_pretrain

python pretrain.py --config_file config/pretrain_config.json --train_data_file $TRAIN_DATA_FILE  --val_data_file $VAL_DATA_FILE --vocab_file $VOCAB_PATH
```
See [params.py](../params.py) and [config/pretrain_config.json](../config/pretrain_config.json) about the others parameters


### Fine-tuning
```bash
TASK=bias_classification
TRAIN_DATA_FILE=bias/data/train.csv
VAL_DATA_FILE=bias/data/todo.csv
VOCAB_PATH=bias/data/simple_vocab3.txt

python classify.py --config_file config/classif_config.json --task $TASK --train_data_file $TRAIN_DATA_FILE --val_data_file $VAL_DATA_FILE --vocab_file $VOCAB_PATH 
```
See [params.py](../params.py) and [config/classif_config.json](../config/classif_config.json) about the others parameters

### Evaluation of the trained Classifier
```bash
TASK=bias_classification
VAL_DATA_FILE=bias/data/todo.csv
VOCAB_PATH=bias/data/simple_vocab3.txt
RELOAD_MODEL=/content/bert_classification/classification/demo/best_val_acc.pt
EVAL_ONLY=True
EXP_ID=eval

python classify.py --config_file config/classif_config.json --reload_model $RELOAD_MODEL --eval_only $EVAL_ONLY --task $TASK --val_data_file $VAL_DATA_FILE --vocab_file $VOCAB_PATH --exp_id $EXP_ID
```