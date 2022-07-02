cd $(dirname $(dirname $0))
source activate xai
export PYTHONPATH=${PYTHONPATH}:/home/yila22/prj:/mnt/yixin/

## experiment hyperp
exp_name="hyper-searching-0703"
dataset=(hate rotten_tomatoes  imdb sst emoji  \
                sentiment  stance_abortion  stance_atheism  stance_climate  stance_feminist  \
                stance_hillary)
n_iters=40
K=7

## something you need to decide
gpu=(2 4 5 6)
allow_gpu_memory_threshold=5000
gpuu_threshold=90
sleep_time_after_loading_task=20s
all_full_sleep_time=20s
total_task=10

## something can be auto decided
task_counter=0
gpunum=${#gpu[@]}
START_TIME=`date +%s`

# main running
for seed in 10 ; do
for model in simple-rnn lstm; do
for pgd_radius in 0.005 0.01 0.02;do
for x_pgd_radius in 0.01; do
for datasetid in 3 0 4 5 6 7 8 9 10; do
for lambda_1 in 0 1e-4 1; do
  for lambda_2 in 0 1e-4 1; do
# find suitable gpu
i=0 # we search from the first gpu
while true; do
    gpu_id=${gpu[$i]}
#    nvidia-smi --query-gpu=utilization.gpu  --format=csv -i 2 | grep -Eo "[0-9]+"
    gpu_u=$(nvidia-smi --query-gpu=utilization.gpu  --format=csv -i $gpu_id | grep -Eo "[0-9]+")
    free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $gpu_id | grep -Eo "[0-9]+")
    if [[ $free_mem -lt $allow_gpu_memory_threshold || $gpu_u -ge ${gpuu_threshold} ]]; then
        i=`expr $i + 1`
        i=`expr $i % $gpunum`
        echo "gpu id ${gpu[$i]} is full loaded, skip"
        if [ "$i" == "0" ]; then
            sleep ${all_full_sleep_time}
            echo "all the gpus are full, sleep 1m"
        fi
    else
        break
    fi
done

END_TIME=`date +%s`
seconds=`expr $END_TIME - $START_TIME`
hour=$(( $seconds/3600 ))
min=$(( ($seconds-${hour}*3600)/60 ))
sec=$(( $seconds-${hour}*3600-${min}*60 ))
HMS=`echo ${hour}:${min}:${sec}`
#echo "Time have elapsed ${HMS}"
task_counter=`expr $task_counter + 1`
TOTAL_TIME_PROX="expr $EXECUTING_TIME / $task_counter"
TOTAL_TIME_PROX="expr $TOTAL_TIME_PROX\* $total_task"
LEFT_TIME_PROX="expr $TOTAL_TIME_PROX - $EXECUTING_TIME"

total_hour=$(( $LEFT_TIME_PROX/3600 ))
total_min=$(( ($LEFT_TIME_PROX-${total_hour}*3600)/60 ))
total_sec=$(( $LEFT_TIME_PROX-${total_hour}*3600-${total_min}*60 ))
HMS_PROX=`echo ${total_hour}:${total_min}:${total_sec}`

echo "EXECUTING_TIME: $HMS, PROX LEFT TIME $HMS_PROX, FINISH TASKS/TOTAL TASKS: $task_counter/$total_task"

gpu_id=${gpu[$i]}
free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $gpu_id | grep -Eo "[0-9]+")
gpu_u=$(nvidia-smi --query-gpu=utilization.gpu  --format=csv -i $gpu_id | grep -Eo "[0-9]+")
export CUDA_VISIBLE_DEVICES=$gpu_id
echo "use gpu id is ${gpu[$i]}, free memory is ${free_mem}, it utilization is ${gpu_u}%"

    com="python train.py --dataset ${dataset[$datasetid]} --data_dir . --output_dir test_ours_outputs_seed/ \
    --encoder $model --ours --n_iters $n_iters \
      --exp_name $exp_name --lambda_1 $lambda_1 --lambda_2 $lambda_2 --pgd_radius $pgd_radius --x_pgd_radius $x_pgd_radius \
      --K $K  --seed $seed"
    echo $com
    nohup $com > ./logs/$exp_name-$RANDOM.log 2>&1 &
    echo "sleep for ${sleep_time_after_loading_task} to wait the task loaded"
    sleep  ${sleep_time_after_loading_task} # you need to wait for this task fully loaded so that gpu stat changes!
  done;
done;
done;
done;
done;
done;
done;

#
#for seed in 10; do
#for model in simple-rnn lstm; do
#for pgd_radius in 0.005 0.01 0.02;do
#for x_pgd_radius in 0.01; do
#for datasetid in 2; do
#for lambda_1 in 0 1e-4 1e-3 1e-2 1e-1 1; do
#  for lambda_2 in 0 1e-4 1e-3 1e-2 1e-1 1; do
## find suitable gpu
#i=0 # we search from the first gpu
#while true; do
#    gpu_id=${gpu[$i]}
##    nvidia-smi --query-gpu=utilization.gpu  --format=csv -i 2 | grep -Eo "[0-9]+"
#    gpu_u=$(nvidia-smi --query-gpu=utilization.gpu  --format=csv -i $gpu_id | grep -Eo "[0-9]+")
#    free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $gpu_id | grep -Eo "[0-9]+")
#    if [[ $free_mem -lt $task_load || $gpu_u -ge ${gpuu_threshold} ]]; then
#        i=`expr $i + 1`
#        i=`expr $i % $gpunum`
#        echo "gpu id ${gpu[$i]} is full loaded, skip"
#        if [ "$i" == "0" ]; then
#            sleep 1m
#            echo "all the gpus are full, sleep 1m"
#        fi
#    else
#        break
#    fi
#done
#gpu_id=${gpu[$i]}
#free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $gpu_id | grep -Eo "[0-9]+")
#gpu_u=$(nvidia-smi --query-gpu=utilization.gpu  --format=csv -i $gpu_id | grep -Eo "[0-9]+")
#export CUDA_VISIBLE_DEVICES=$gpu_id
#echo "use gpu id is ${gpu[$i]}, free memory is ${free_mem}, it utilization is ${gpu_u}%"
#    com="python train.py --dataset ${dataset[$datasetid]} --data_dir . --output_dir test_ours_outputs_seed/ \
#    --encoder $model --ours --n_iters $n_iters \
#      --exp_name $exp_name --lambda_1 $lambda_1 --lambda_2 $lambda_2 --pgd_radius $pgd_radius --x_pgd_radius $x_pgd_radius \
#      --K $K  --seed $seed"
#    nohup $com > ./logs/$exp_name-$RANDOM.log 2>&1 &
##     $com
#    echo "sleep for ${up_task_time} to wait the task loaded"
#    sleep  ${up_task_time} # you need to wait for this task fully loaded so that gpu stat changes!
#  done;
#done;
#done;
#done;
#done;
#done;
#done;


#cd $(dirname $(dirname $0))
#source activate xai
##export PYTHONPATH=/home/yila22/prj
#export PYTHONPATH=${PYTHONPATH}:/home/yila22/prj:/mnt/yixin/
#exp_name="find-best-hyperparameters-v3"
#
#dataset=(hate rotten_tomatoes  imdb sst emoji  \
#                sentiment  stance_abortion  stance_atheism  stance_climate  stance_feminist  \
#                stance_hillary)
#gpu=(1 2 3 4 5 6 )
#gpunum=${#gpu[@]}
#i=0 # gpu pointer
#j=0
#n_iters=40
#K=7
#
#for model in simple-rnn lstm; do
#for pgd_radius in 0.01;do
#for x_pgd_radius in 0.01; do
#for datasetid in 0 1 3 4 5 6 7 8 9 10; do
##for datasetid in 2 3; do
##for lambda_1 in 1; do
##for lambda_2 in 1e-4; do
#for lambda_1 in 0 1e-4 1e-3 1e-2 1e-1 1; do
#  for lambda_2 in 0 1e-4 1e-3 1e-2 1e-1 1; do
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
##    if [ "$j" == "0" ];then
##      sleep 15m
##    fi
#  done;
#done;
#done;
#done;
#done;
#done;

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