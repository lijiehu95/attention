

source activate xai
export PYTHONPATH=/home/yila22/prj
export CUDA_VISIBLE_DEVICES='5'
exp_name="ours-debug-v6"

dataset=(sst imdb hate offensive rotten_tomatoes)
golddir=( "/home/yila22/prj/attention/test_outputs/sst/lstm+tanh/Wed_Jun_22_11:01:35_2022/" \
  "/home/yila22/prj/attention/test_outputs/imdb/lstm+tanh/Wed_Jun_22_11:01:34_2022/" \
  "/home/yila22/prj/attention/test_outputs/hate/lstm+tanh/Tue_Jun_21_15:33:19_2022/" \
  "/home/yila22/prj/attention/test_outputs/offensive/lstm+tanh/Wed_Jun_22_11:01:36_2022/" \
  "/home/yila22/prj/attention/test_outputs/rotten_tomatoes/lstm+tanh/Wed_Jun_22_11:00:16_2022/" \
)

gpu=(1 2 7 8)
gpunum=4
i=0 # gpu pointer

n_iters=10
K=7
for pgd_radius in 0.001 0.01 0.1;do
for x_pgd_radius in 0.0005 0.001 0.01 0.1; do
for datasetid in 0; do
for lambda_1 in 1e-3 1e-2 1e-1 1 10; do
  for lambda_2 in 1e-3 1e-2 1e-1 1 10; do
    # in the for loop
   i=`expr $i % $gpunum`
   export CUDA_VISIBLE_DEVICES=${gpu[$i]}
   echo "use gpu id is ${gpu[$i]}"

   # your command here
    com="python train.py --dataset ${dataset[$datasetid]} --data_dir . --output_dir test_adv_outputs/ \
      --encoder lstm --ours --gold_label_dir ${golddir[$datasetid]} --n_iters $n_iters \
        --exp_name $exp_name --lambda_1 $lambda_1 --lambda_2 $lambda_2 --pgd_radius $pgd_radius --x_pgd_radius $x_pgd_radius \
        --K $K"
    # nohup $com > ./logs/$exp_name-$RANDOM.log 2>&1 &
    $com

    i=`expr $i + 1`
done;done;done;done;done;