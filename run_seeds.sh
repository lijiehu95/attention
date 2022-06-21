
source activate xai
export PYTHONPATH=/home/yila22/prj
export CUDA_VISIBLE_DEVICES='2,8'
exp_name="debug-seed"


for dataset in sst imdb hate emotion;do
for seed in 50 257 500231 100078 12504 90754789 8988812 2; do
python train.py --dataset $dataset --data_dir . --output_dir test_outputs/dataset-${dataset}-seed-${seed}/ --attention tanh --encoder lstm --seed $seed \
  --exp_name $exp_name
done;done