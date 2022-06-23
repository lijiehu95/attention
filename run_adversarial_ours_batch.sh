

source activate xai
export PYTHONPATH=/home/yila22/prj
export CUDA_VISIBLE_DEVICES='5'
exp_name="ours-debug-v6"

dataset=(sst imdb hate rotten_tomatoes)
golddir=( "/home/yila22/prj/attention/test_outputs/sst/lstm+tanh/Wed_Jun_22_11:01:35_2022/" \
  "/home/yila22/prj/attention/test_outputs/imdb/lstm+tanh/Wed_Jun_22_11:01:34_2022/" \
  "/home/yila22/prj/attention/test_outputs/hate/lstm+tanh/Tue_Jun_21_15:33:19_2022/" \
  "/home/yila22/prj/attention/test_outputs/rotten_tomatoes/lstm+tanh/Wed_Jun_22_11:00:16_2022/" \
)

gpu=(3 4 5 6 7 8 9)
gpunum=7
i=0 # gpu pointer
j=0
n_iters=40
K=7
for pgd_radius in 0.001 0.01 0.1;do
for x_pgd_radius in 0.001 0.01 0.1; do
for datasetid in 0 1 2 3; do
for lambda_1 in 0 0.01 0.1 1; do
  for lambda_2 in 0 0.01 0.1 1; do
   i=`expr $i % $gpunum`
   export CUDA_VISIBLE_DEVICES=${gpu[$i]}
   echo "use gpu id is ${gpu[$i]}"
   com="python train.py --dataset ${dataset[$datasetid]} --data_dir . --output_dir test_adv_outputs/ \
    --encoder lstm --ours --gold_label_dir ${golddir[$datasetid]} --n_iters $n_iters \
      --exp_name $exp_name --lambda_1 $lambda_1 --lambda_2 $lambda_2 --pgd_radius $pgd_radius --x_pgd_radius $x_pgd_radius \
      --K $K"
    nohup $com > ./logs/$exp_name-$RANDOM.log 2>&1 &
    # $com
    i=`expr $i + 1`
    j=`expr $j + 1`
    j=`expr $j % $gpunum`
    if [ "$j" == "0" ];then
      sleep 20m
    fi
  done;
done;
done;
done;
done;