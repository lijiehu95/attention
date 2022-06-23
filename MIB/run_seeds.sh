
source activate xai
export PYTHONPATH=/mnt/yixin/
export CUDA_VISIBLE_DEVICES='0'
exp_name="seed-v2"

n_iters=40
for dataset in sst imdb hate rotten_tomatoes;do
for seed in 50 257 ; do
  com="python train.py --dataset $dataset --data_dir . --output_dir test_outputs/ --attention tanh --encoder lstm --seed $seed \
  --exp_name $exp_name --n_iters $n_iters"
  nohup $com > $exp_name-$dataset-$seed.log 2>&1 &
done;done