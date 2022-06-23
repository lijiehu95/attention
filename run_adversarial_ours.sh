

source activate xai
export PYTHONPATH=/home/yila22/prj
export CUDA_VISIBLE_DEVICES='7'
exp_name="ours-final-debug-624"

dataset=(sst imdb hate offensive rotten_tomatoes)
golddir=( "/home/yila22/prj/attention/test_outputs/sst/lstm+tanh/Wed_Jun_22_11:01:35_2022/" \
  "/home/yila22/prj/attention/test_outputs/imdb/lstm+tanh/Wed_Jun_22_11:01:34_2022/" \
  "/home/yila22/prj/attention/test_outputs/hate/lstm+tanh/Tue_Jun_21_15:33:19_2022/" \
  "/home/yila22/prj/attention/test_outputs/offensive/lstm+tanh/Wed_Jun_22_11:01:36_2022/" \
  "/home/yila22/prj/attention/test_outputs/rotten_tomatoes/lstm+tanh/Wed_Jun_22_11:00:16_2022/" \
)

n_iters=40
K=7
for pgd_radius in 0.01;do
for x_pgd_radius in 0.001; do
for datasetid in 0 1 2 3 4 ; do
for lambda_1 in 1; do
  for lambda_2 in 1; do
    com="python train.py --dataset ${dataset[$datasetid]} --data_dir . --output_dir test_adv_outputs/ \
      --encoder lstm --ours --gold_label_dir ${golddir[$datasetid]} --n_iters $n_iters \
        --exp_name $exp_name --lambda_1 $lambda_1 --lambda_2 $lambda_2 --pgd_radius $pgd_radius --x_pgd_radius $x_pgd_radius \
        --K $K"
    $com
done;done;done;done;done;