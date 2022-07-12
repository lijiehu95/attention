exp_name="baseline"
export CUDA_VISIBLE_DEVICES=0
for model in simple-rnn lstm; do
for dataset in hate rotten_tomatoes  imdb sst;do
#                emoji sentiment  stance_abortion  stance_atheism  stance_climate  stance_feminist  \
#                stance_hillary \
#                ; do
python train.py --dataset $dataset --data_dir . --output_dir test_baseline/ --attention tanh \
    --encoder $model --exp_name $exp_name --baseline
done;done;

#cd $(dirname $(dirname $0))
#source activate xai
#export PYTHONPATH=${PYTHONPATH}:/home/yila22/prj:/mnt/yixin/
#
#
### experiment hyperp
#exp_name="run-baseline"
#dataset=(hate rotten_tomatoes  imdb sst emoji  \
#                sentiment  stance_abortion  stance_atheism  stance_climate  stance_feminist  \
#                stance_hillary)
#n_iters=40
#K=7
#
### something you need to decide
#gpu=(2 3 4 5 6 7 8 9)
#allow_gpu_memory_threshold=5000
#gpuu_threshold=90
#sleep_time_after_loading_task=20s
#all_full_sleep_time=20s
#total_task=540
#
### something can be auto decided
#task_counter=0
#gpunum=${#gpu[@]}
#START_TIME=`date +%s`
#
## main running
#for seed in 51235 ; do
#for model in simple-rnn lstm; do
#for datasetid in 0 1 2 3 4 5 6 7 8 9 10; do
## find suitable gpu
#i=0 # we search from the first gpu
#while true; do
#    gpu_id=${gpu[$i]}
##    nvidia-smi --query-gpu=utilization.gpu  --format=csv -i 2 | grep -Eo "[0-9]+"
#    gpu_u=$(nvidia-smi --query-gpu=utilization.gpu  --format=csv -i $gpu_id | grep -Eo "[0-9]+")
#    free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $gpu_id | grep -Eo "[0-9]+")
#    if [[ $free_mem -lt $allow_gpu_memory_threshold || $gpu_u -ge ${gpuu_threshold} ]]; then
#        i=`expr $i + 1`
#        i=`expr $i % $gpunum`
#        echo "gpu id ${gpu[$i]} is full loaded, skip"
#        if [ "$i" == "0" ]; then
#            sleep ${all_full_sleep_time}
#            echo "all the gpus are full, sleep 1m"
#        fi
#    else
#        break
#    fi
#done
#
#END_TIME=`date +%s`
#seconds=`expr $END_TIME - $START_TIME`
#hour=$(( $seconds/3600 ))
#min=$(( ($seconds-${hour}*3600)/60 ))
#sec=$(( $seconds-${hour}*3600-${min}*60 ))
#HMS=`echo ${hour}h:${min}m:${sec}s`
##echo "Time have elapsed ${HMS}"
#task_counter=`expr $task_counter + 1`
#
#echo "EXECUTING_TIME: $HMS, FINISH TASKS/TOTAL TASKS: $task_counter/$total_task"
#
#gpu_id=${gpu[$i]}
#free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $gpu_id | grep -Eo "[0-9]+")
#gpu_u=$(nvidia-smi --query-gpu=utilization.gpu  --format=csv -i $gpu_id | grep -Eo "[0-9]+")
#export CUDA_VISIBLE_DEVICES=$gpu_id
#echo "use gpu id is ${gpu[$i]}, free memory is ${free_mem}, it utilization is ${gpu_u}%"
#
##    com="python train.py --dataset ${dataset[$datasetid]} --data_dir . --output_dir test_ours_outputs_seed/ \
##    --encoder $model --ours --n_iters $n_iters \
##      --exp_name $exp_name --lambda_1 $lambda_1 --lambda_2 $lambda_2 --pgd_radius $pgd_radius --x_pgd_radius $x_pgd_radius \
##      --K $K  --seed $seed"
#  com="python train.py --dataset ${dataset[$datasetid]} --data_dir . --output_dir train_baseline/ --attention tanh \
#    --encoder $model --exp_name $exp_name --baseline"
#    echo $com
#    echo ==========================================================================================
#    nohup $com > ./logs/$exp_name-$RANDOM.log 2>&1 &
#    echo "sleep for ${sleep_time_after_loading_task} to wait the task loaded"
#    sleep  ${sleep_time_after_loading_task} # you need to wait for this task fully loaded so that gpu stat changes!
#  done;
#done;
#done;
#done;
#done;
#done;
#done;
