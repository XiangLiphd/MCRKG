#!/bin/bash
dt=`date '+%Y%m%d_%H%M%S'`

dataset="ele_data2"
model="chinese-roberta-wwm"
model_path="/home/inspur/lx/models/chinese-reberta-wwm"
shift
shift
args=$@

elr="1e-5"
dlr="1e-3"
bs=128
mbs=2
n_epochs=200
num_relation=4

seed=42
k=4 #num of gnn layers
gnndim=200

echo "***** hyperparameters *****"
echo "dataset: $dataset"
echo "enc_name: $model"
echo "batch_size: $bs"
echo "learning_rate: elr $elr dlr $dlr"
echo "gnn: dim $gnndim layer $k"
echo "seed: $seed"
echo "******************************"

save_dir_pref='saved_models'
mkdir -p $save_dir_pref
mkdir -p logs


###### Training ######
python -u mcrkg.py \
    --dataset $dataset \
    --encoder $model \
    --model_path $model_path \
    -k $k \
    --gnn_dim $gnndim \
    -elr $elr \
    -dlr $dlr \
    -bs $bs \
    -mbs $mbs \
    --seed $seed \
    --num_relation $num_relation \
    --n_epochs $n_epochs \
    --max_epochs_before_stop 10 \
    --train_adj  data/${dataset}/graph/train.graph.adj.pk \
    --dev_adj  data/${dataset}/graph/dev.graph.adj.pk \
    --test_adj  data/${dataset}/graph/test.graph.adj.pk \
    --train_statements  /home/inspur/lx/mcrkg/data/ele_data2/statement/train.statement.jsonl \
    --dev_statements  /home/inspur/lx/mcrkg/data/ele_data2/statement/dev.statement.jsonl \
    --test_statements  /home/inspur/lx/mcrkg/data/ele_data2/statement/test.statement.jsonl \
    --use_cache  true \
    --case_db_path /home/inspur/lx/mcrkg/data/case_db.jsonl \
    --pool_type mean \
    --num_pos 5 \
    --num_neg 5 \
    --inhouse false \
    --save_dir ${save_dir_pref}/${dataset}/enc-${model}__k${k}__gnndim${gnndim}__bs${bs}__seed${seed}__${dt} \
    >  logs/train_${dataset}__enc-${model}__k${k}__gnndim${gnndim}__bs${bs}__seed${seed}__${dt}.log

    # --save_model false \
