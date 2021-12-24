## librairies
```bash
pip install -r requirements.txt
import nltk
nltk.download('punkt')
```

## Data preprocessings
```bash
datapath=/content
references_files=""
data_columns=content,labels
save_to=/content

python preprocessed_data.py -f ${datapath}/data_train.csv,${datapath}/data_val.csv,${datapath}/data_test.csv -rf $references_files -dc $data_columns  -st $save_to
```

## Rename files
```bash
for data_type in train test val; do 
    mv ${save_to}/data_${data_type}_csv.csv ${save_to}/data_${data_type}.csv
done
```

## Pretrain
```bash
load_from_checkpoint=None
#load_from_checkpoint=/content/pretrain/1
eval_only=False
task=pretrain
! . main.sh $dump_path $data_columns $task $load_from_checkpoint $eval_only
```

# Eval pretrain
```bash
load_from_checkpoint=/content/pretrain/1
eval_only=True
task=pretrain
. main.sh $dump_path $data_columns $task $load_from_checkpoint $eval_only
```

## Debias
```bash
load_from_checkpoint=/content/pretrain/1
#load_from_checkpoint=/content/debias/1
eval_only=False
task=debias
. main.sh $dump_path $data_columns $task $load_from_checkpoint $eval_only
```

## Eval Debias
```bash
load_from_checkpoint=/content/debias/1
eval_only=True
task=debias
. main.sh $dump_path $data_columns $task $load_from_checkpoint $eval_only
```

```
@misc{wang2019controllable,
      title={Controllable Unsupervised Text Attribute Transfer via Editing Entangled Latent Representation}, 
      author={Ke Wang and Hang Hua and Xiaojun Wan},
      year={2019},
      eprint={1905.12926},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```