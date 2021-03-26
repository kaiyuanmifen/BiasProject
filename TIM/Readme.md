## Requirements

Python > 3.6, tqdm, tensorflow (for loading checkpoint file)

```bash
git clone -b tim https://github.com/kaiyuanmifen/BiasProject
cd BiasProject/TIM
pip install -r requirements.txt
```

See [bias/Readme.md](bias/Readme.md) for bias classification.

## Overview

This contains 9 python files.
- [`tokenization1.py`](./tokenization1.py) : Tokenizers adopted from the original Google BERT's code (option 1 : default option)
- [`tokenization2.py`](./tokenization2.py) : https://pypi.org/project/tokenizers/ (option 2)
- [`tokenization3.py`](./tokenization3.py) : tf (option 3)
- [`checkpoint.py`](./checkpoint.py) : Functions to load a model from tensorflow's checkpoint file
- [`models.py`](./models.py) : Model classes for a general transformer and TIM
- [`dataset.py`](./dataset.py) : dataset for pretraining (option 1 in [`pretrain.py`](./pretrain.py) : deprecated) and fine-tuning
- [`dataset2.py`](./dataset2.py) : dataset for pretraining (option 2 in [`pretrain.py`](./pretrain.py) : default option)
- [`vocab.py`](./vocab.py) : todo
- [`params.py`](./params.py) : Generate a parameters parser.
- [`slurm.py`](./slurm.py) : Handle single and multi-GPU / multi-node / SLURM jobs.
- [`logger.py`](./logger.py) : Create a logger, Use a different log file for each process, create file handler and set level to debug
- [`optim.py`](./optim.py) : A custom optimizer (BertAdam class) adopted from Hugging Face's code
- [`train.py`](./train.py) : A helper class for training and evaluation
- [`utils.py`](./utils.py) : Several utility functions
- [`pretrain.py`](./pretrain.py) : An example code for pre-training transformer
- [`classify.py`](./classify.py) : An example code for fine-tuning using pre-trained transformer

## Example Usage

### Pre-training Transformer

#### Option 1 (dataset.py : deprecated) : inspired by [pytorchic-bert](https://github.com/dhlee347/pytorchic-bert)
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

#### Option 2 (dataset2.py) : inspired by [BERT-pytorch](https://github.com/codertimo/BERT-pytorch)
Input file format : one sentence per line 
```
Welcome to the \t the jungle\n
I can stay \t here all night\n
```
or tokenized corpus (tokenization is not in package)
```
Wel_ _come _to _the \t _the _jungle\n
_I _can _stay \t _here _all _night\n
```

This code snippet can be used to switch from a file in option 1 format to a file in option 2 format.
```python
data_file = "data/pretrain.txt"
data_file2 = "data/pretrain2.txt"
with open(data_file) as f:
    data = f.readlines()

def process(x):
    x = x.split()
    l = len(x)//2 # todo : random
    return " ".join(x[:l] + ["\t"] + x[l:])

data = [process(x) + "\n" for x in data if x != "\n"]

with open(data_file2, "w") as f:
    f.writelines(data)
```

Usage (manually change the default options in [`pretrain.py`](./pretrain.py) if necessary.) :
```bash
TRAIN_DATA_FILE=data/pretrain.txt
VAL_DATA_FILE=data/todo.txt
VOCAB_PATH=data/vocab.txt

python pretrain.py --config_file config/pretrain_config.json --train_data_file $TRAIN_DATA_FILE  --val_data_file $VAL_DATA_FILE --vocab_file $VOCAB_PATH
```
See [params.py](./params.py) and [config/pretrain_config.json](./config/pretrain_config.json) about the others parameters

### Fine-tuning (MRPC, sentiment_analysis, MNLI) Classifier with Pre-trained Transformer
 
```bash
TASK=sentiment_analysis
TRAIN_DATA_FILE=data/train.csv
VAL_DATA_FILE=data/todo.csv
VOCAB_PATH=data/vocab.txt

python classify.py --config_file config/classif_config.json --task $TASK --train_data_file $TRAIN_DATA_FILE --val_data_file $VAL_DATA_FILE --vocab_file $VOCAB_PATH 
```
See [params.py](./params.py) and [config/classif_config.json](.config/classif_config.json) about the others parameters

### Evaluation of the trained Classifier
```bash
TASK=sentiment_analysis
VAL_DATA_FILE=data/eval.csv
VOCAB_PATH=data/vocab.txt
RELOAD_MODEL=/content/bert_classification/classification/demo/best_val_acc.pt
EVAL_ONLY=True
EXP_ID=eval

python classify.py --config_file config/classif_config.json --reload_model $RELOAD_MODEL --eval_only $EVAL_ONLY --task $TASK --val_data_file $VAL_DATA_FILE --vocab_file $VOCAB_PATH --exp_id $EXP_ID
```

### References

[references.py](./references.py)