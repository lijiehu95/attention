source activate xai
export PYTHONPATH=/home/yila22/prj
export CUDA_VISIBLE_DEVICES='2'
exp_name="seed-v1"

n_iters=40
for model in simple-rnn; do
for dataset in imdb hate rotten_tomatoes;do
for seed in 50 257 500231 2; do
 com="python train.py --dataset $dataset --data_dir . --output_dir seed_output/ --attention tanh --encoder $model --seed $seed \
  --exp_name $exp_name --n_iters $n_iters"
#  nohup $com > $exp_name-$dataset-$seed.log 2>&1 &
  $com
done;done;done