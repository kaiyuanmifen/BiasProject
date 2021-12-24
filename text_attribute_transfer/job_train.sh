#!/bin/bash
#SBATCH --job-name=training
#SBATCH --gres=gpu:2              # Number of GPUs (per node)
#SBATCH --mem=85G               # memory (per node)
#SBATCH --time=0-12:00            # time (DD-HH:MM)

###load environment 
#module load anaconda/3
module load cuda/10.1
source ../at/bin/activate

dump_path=../dump_path
data_columns=content,labels

load_from_checkpoint=None
#load_from_checkpoint=${dump_path}/pretrain/1
#load_from_checkpoint=${dump_path}/content/debias/1

eval_only=False
#eval_only=True

task=pretrain
#task=debias

filename=main.sh
chmod +x $filename
cat $filename | tr -d '\r' > $filename.new && rm $filename && mv $filename.new $filename 

#sbatch . main.sh $dump_path $data_columns $task $load_from_checkpoint $eval_only
. main.sh $dump_path $data_columns $task $load_from_checkpoint $eval_only