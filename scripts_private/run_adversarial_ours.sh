cd $(dirname $(dirname $0))
source activate xai
#export PYTHONPATH=/home/yila22/prj
export PYTHONPATH=${PYTHONPATH}:/home/yila22/prj:/mnt/yixin/
exp_name="find-best-hyperparameters-v1"

dataset=(hate rotten_tomatoes  imdb sst emoji  \
                sentiment  stance_abortion  stance_atheism  stance_climate  stance_feminist  \
                stance_hillary)
gpu=(2 3 4 5 6 7 8 9)
gpunum=${#gpu[@]}
i=0 # gpu pointer
j=0
n_iters=40
K=7

for model in lstm; do
for pgd_radius in 0.001 0.005 0.01 0.05 0.1;do
for x_pgd_radius in 0.001 0.005 0.01 0.05 0.1; do
for datasetid in 1; do
for lambda_1 in 0 1e-3 1e-2 1e-1 1; do
  for lambda_2 in 0 1e-3 1e-2 1e-1 1; do
    i=`expr $i % $gpunum`
    export CUDA_VISIBLE_DEVICES=${gpu[$i]}
    echo "use gpu id is ${gpu[$i]}"
    com="python train.py --dataset ${dataset[$datasetid]} --data_dir . --output_dir test_ours_outputs/ \
    --encoder $model --ours --n_iters $n_iters \
      --exp_name $exp_name --lambda_1 $lambda_1 --lambda_2 $lambda_2 --pgd_radius $pgd_radius --x_pgd_radius $x_pgd_radius \
      --K $K  "
    nohup $com > ./logs/$exp_name-$RANDOM.log 2>&1 &
#     $com
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
done;

#
#for model in lstm simple-rnn; do
#for pgd_radius in 0.1;do
#for x_pgd_radius in 0.001 0.005 0.01 0.05 0.1; do
#for datasetid in 0 1 2 3 4 5 6; do
#for lambda_1 in 0 1e-2 1e-1 1; do
#  for lambda_2 in 0 1e-2 1e-1 1; do
#    i=`expr $i % $gpunum`
#    export CUDA_VISIBLE_DEVICES=${gpu[$i]}
#    echo "use gpu id is ${gpu[$i]}"
#    com="python train.py --dataset ${dataset[$datasetid]} --data_dir . --output_dir test_ours_outputs/ \
#    --encoder $model --ours --n_iters $n_iters \
#      --exp_name $exp_name --lambda_1 $lambda_1 --lambda_2 $lambda_2 --pgd_radius $pgd_radius --x_pgd_radius $x_pgd_radius \
#      --K $K  "
##    nohup $com > ./logs/$exp_name-$RANDOM.log 2>&1 &
#     $com
#    i=`expr $i + 1`
#    j=`expr $j + 1`
#    j=`expr $j % $gpunum`
#  done;
#done;
#done;
#done;
#done;
#done;
#
#for model in lstm simple-rnn; do
#for pgd_radius in 0.01;do
#for x_pgd_radius in 0.001 0.005 0.01 0.05 0.1; do
#for datasetid in 0 1 2 3 4 5 6; do
#for lambda_1 in 0 1e-2 1e-1 1; do
#  for lambda_2 in 0 1e-2 1e-1 1; do
#    i=`expr $i % $gpunum`
#    export CUDA_VISIBLE_DEVICES=${gpu[$i]}
#    echo "use gpu id is ${gpu[$i]}"
#    com="python train.py --dataset ${dataset[$datasetid]} --data_dir . --output_dir test_ours_outputs/ \
#    --encoder $model --ours --n_iters $n_iters \
#      --exp_name $exp_name --lambda_1 $lambda_1 --lambda_2 $lambda_2 --pgd_radius $pgd_radius --x_pgd_radius $x_pgd_radius \
#      --K $K  "
##    nohup $com > ./logs/$exp_name-$RANDOM.log 2>&1 &
#     $com
#    i=`expr $i + 1`
#    j=`expr $j + 1`
#    j=`expr $j % $gpunum`
#  done;
#done;
#done;
#done;
#done;
#done;


#
#dataset=(sst imdb hate rotten_tomatoes)
#golddir=( "/home/yila22/prj/attention/test_outputs/sst/simple-rnn+tanh/Thu_Jun_23_16:01:58_2022" \
#  "/home/yila22/prj/attention/test_outputs/imdb/simple-rnn+tanh/Thu_Jun_23_16:02:05_2022" \
#  "/home/yila22/prj/attention/test_outputs/hate/simple-rnn+tanh/Thu_Jun_23_16:01:56_2022" \
#  "/home/yila22/prj/attention/test_outputs/rotten_tomatoes/simple-rnn+tanh/Thu_Jun_23_16:01:59_2022/" \
#)
##gpu=(0 1 2 3 4 5 6 7 8 9)
##gpunum=10
##i=0 # gpu pointer
#j=0
#n_iters=40
#K=7
#for pgd_radius in 0.001;do
#for x_pgd_radius in 0.005; do
#for datasetid in 1; do
#for lambda_1 in 1; do
#  for lambda_2 in 1e-1; do
#   i=`expr $i % $gpunum`
#   export CUDA_VISIBLE_DEVICES=${gpu[$i]}
#   echo "use gpu id is ${gpu[$i]}"
#   com="python train.py --dataset ${dataset[$datasetid]} --data_dir . --output_dir test_adv_outputs/ \
#    --encoder simple-rnn --ours --gold_label_dir ${golddir[$datasetid]} --n_iters $n_iters \
#      --exp_name $exp_name --lambda_1 $lambda_1 --lambda_2 $lambda_2 --pgd_radius $pgd_radius --x_pgd_radius $x_pgd_radius \
#      --K $K"
##    nohup $com > ./logs/$exp_name-$RANDOM.log 2>&1 &
#     $com
#    i=`expr $i + 1`
#    j=`expr $j + 1`
#    j=`expr $j % $gpunum`
##    if [ "$j" == "0" ];then
##      sleep 20m
##    fi
#  done;
#done;
#done;
#done;
#done;