source activate xai
export PYTHONPATH=/home/yila22/prj
export CUDA_VISIBLE_DEVICES='9'
exp_name="baseline-v2"

#datasets=("offensive" "irony" "rotten_tomatoes")

for dataset in  "irony" "rotten_tomatoes"; do
  com="python train.py --dataset $dataset --data_dir . --output_dir test_outputs/ --attention tanh --encoder lstm --exp_name $exp_name"
  $com
#  nohup $com > null 2>&1 &
done;
export CUDA_VISIBLE_DEVICES='4'
exp_name="freeze-att-lstm-v2"
for dataset in  "irony" "rotten_tomatoes";do
  com="python train.py --dataset $dataset --data_dir . --output_dir test_outputs/ --attention frozen --encoder lstm  \
    --exp_name $exp_name"
  $com
#  nohup $com > null 2>&1 &
done;

