#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
dt=`date '+%Y%m%d_%H%M%S'`


dataset="ele_data"
model='chinese-roberta-wwm'
shift
shift
args=$@


echo "******************************"
echo "dataset: $dataset"
echo "******************************"

save_dir_pref='saved_models'
mkdir -p $save_dir_pref

###### Eval ######
python3 -u mcrkg.py --dataset $dataset \
      --train_adj data/${dataset}/graph/train.graph.adj.pk \
      --dev_adj   data/${dataset}/graph/dev.graph.adj.pk \
      --test_adj  data/${dataset}/graph/test.graph.adj.pk \
      --train_statements data/${dataset}/statement/train.statement.jsonl \
      --dev_statements   data/${dataset}/statement/dev.statement.jsonl \
      --test_statements  data/${dataset}/statement/test.statement.jsonl \
      --save_model \
      --save_dir saved_models \
      --mode eval_detail \
      --load_model_path saved_models/ele_data/enc-chinese-roberta-wwm__k5__gnndim200__bs128__seed42__20240216_155433/model.pt.59 \
      $args
