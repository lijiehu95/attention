

source activate xai
export PYTHONPATH=/home/yila22/prj
export CUDA_VISIBLE_DEVICES='7'
exp_name="ours-final-v1"

dataset=(sst imdb emotion hate)
golddir=("/home/yila22/prj/attention/test_outputs/sst/lstm+tanh/Mon_Jun__6_13:13:30_2022/" \
  "/home/yila22/prj/attention/test_outputs/imdb/lstm+tanh/Tue_Jun_21_09:49:04_2022" \
  "/home/yila22/prj/attention/test_outputs/emotion/lstm+tanh/Tue_Jun_21_10:40:50_2022" \
  "/home/yila22/prj/attention/test_outputs/hate/lstm+tanh/Tue_Jun_21_10:31:10_2022")

n_iters=40
K=7
for pgd_radius in 0.01 0.05 0.1 0.3;do
for x_pgd_radius in 0.001 0.01 0.1 0.2; do
for datasetid in 0 1 2 3; do
for lambda_1 in 0 1; do
  for lambda_2 in 0 1; do
    python train.py --dataset ${dataset[$datasetid]} --data_dir . --output_dir test_adv_outputs/ \
      --encoder lstm --ours --gold_label_dir ${golddir[$datasetid]} --n_iters $n_iters \
        --exp_name $exp_name --lambda_1 $lambda_1 --lambda_2 $lambda_2 --pgd_radius $pgd_radius --x_pgd_radius $x_pgd_radius \
        --K $K
done;done;done;done;done;