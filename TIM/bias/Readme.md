### Pre-training
```bash
DATA_FILE=data/pretrain.txt
VOCAB_PATH=data/vocab.txt
SAVE_DIR=/content/bert_pretrain

mkdir $SAVE_DIR
mkdir $SAVE_DIR/runs

python pretrain.py --data_file $DATA_FILE --vocab_file $VOCAB_PATH --save_dir $SAVE_DIR 
```
See [pretrain.py](), [bert_base.json]() and [pretrain.json]() about the others parameters


### Fine-tuning
```bash
TASK=bias_classification
DATA_FILE=data/train.csv
VOCAB_PATH=data/vocab.txt
SAVE_DIR=/content/bert_classification
PRETRAIN_FILE=/content/bert_pretrain/model_steps_5.pt

mkdir $SAVE_DIR
mkdir $SAVE_DIR/runs

python classify.py --mode train --task $TASK --data_file $DATA_FILE --vocab_file $VOCAB_PATH --save_dir $SAVE_DIR --pretrain_file $PRETRAIN_FILE
```
See [classify.py](), [bert_base.json]() and [classification.json]() about the others parameters