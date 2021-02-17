## Requirements

Python > 3.6, fire, tqdm, tensorboardx,
tensorflow (for loading checkpoint file)

## Overview

This contains 9 python files.
- [`tokenization1.py`](./tokenization1.py) : Tokenizers adopted from the original Google BERT's code
- [`tokenization2.py`](./tokenization2.py) : https://pypi.org/project/tokenizers/
- [`tokenization3.py`](./tokenization3.py) : tf
- [`checkpoint.py`](./checkpoint.py) : Functions to load a model from tensorflow's checkpoint file
- [`models.py`](./models.py) : Model classes for a general transformer and TIM
- [`dataset.py`](./dataset.py) : todo
- [`optim.py`](./optim.py) : A custom optimizer (BertAdam class) adopted from Hugging Face's code
- [`train.py`](./train.py) : A helper class for training and evaluation
- [`utils.py`](./utils.py) : Several utility functions
- [`pretrain.py`](./pretrain.py) : An example code for pre-training transformer
- [`classify.py`](./classify.py) : An example code for fine-tuning using pre-trained transformer

## Example Usage

### Pre-training Transformer
Input file format :
1. One sentence per line. These should ideally be actual sentences, not entire paragraphs or arbitrary spans of text. (Because we use the sentence boundaries for the "next sentence prediction" task).
2. Blank lines between documents. Document boundaries are needed so that the "next sentence prediction" task doesn't span between documents.
```
Document 1 sentence 1
Document 1 sentence 2
...
Document 1 sentence 45

Document 2 sentence 1
Document 2 sentence 2
...
Document 2 sentence 24
```
Usage :
```bash
DATA_FILE=data/pretrain.txt
VOCAB_PATH=data/vocab.txt
SAVE_DIR=/content/bert_pretrain

mkdir $SAVE_DIR
mkdir $SAVE_DIR/runs

python pretrain.py --config_file config/pretrain_config.json --data_file $DATA_FILE --vocab_file $VOCAB_PATH --save_dir $SAVE_DIR 
```
See [pretrain.py]() and [config/pretrain_config.json]() about the others parameters

### Fine-tuning (MRPC, sentiment_analysis, MNLI) Classifier with Pre-trained Transformer
Download pretrained model [BERT-Base, Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) and
[GLUE Benchmark Datasets]( https://github.com/nyu-mll/GLUE-baselines) 
before fine-tuning.
* make sure that "total_steps" in train_mrpc.json is n_epochs*(num_data/batch_size)
```bash
TASK=sentiment_analysis
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
### Evaluation of the trained Classifier
```bash
TASK=sentiment_analysis
DATA_FILE=data/eval.csv
VOCAB_PATH=data/vocab.txt
SAVE_DIR=/content/bert_classification
MODE=eval
PRETRAIN_FILE=/content/bert_pretrain/model_steps_5.pt

mkdir $SAVE_DIR
mkdir $SAVE_DIR/runs

python classify.py --config_file config/classif_config.json --mode $MODE --task $TASK --data_file $DATA_FILE --vocab_file $VOCAB_PATH --save_dir $SAVE_DIR --pretrain_file $PRETRAIN_FILE
```






