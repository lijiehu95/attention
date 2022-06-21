
source activate xai
export PYTHONPATH=/home/yila22/prj
export CUDA_VISIBLE_DEVICES='2'
exp_name="seed-v1"

n_iters=40
for dataset in sst imdb hate emotion;do
#for dataset in hate emotion;do
for seed in 50 257 500231 2; do
python train.py --dataset $dataset --data_dir . --output_dir test_outputs/dataset-${dataset}-seed-${seed}/ --attention tanh --encoder lstm --seed $seed \
  --exp_name $exp_name --n_iters $n_iters
done;done