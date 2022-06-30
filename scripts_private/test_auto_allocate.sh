id_list=(2 3 4 5 6 7 8 9)
gpunum=${#gpu[@]}
id_idx=0
free_mem=0
task_load=8000
while True; do
    gpu_id=${gpu[$i]}
    free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $gpu_id | grep -Eo "[0-9]+")
    if [ $free_mem -lt $task_load ]; then
        i=`expr $i + 1`
        i=`expr $i % $gpunum`
        echo "gpu id ${gpu[$i]} is full, free memory isn't less than ${task_load}, skip"
        if [ "$i" == "0" ]; then
            sleep 1m
            echo "all the gpus are full, sleep 1m"
        fi
    else
        break
    fi
done
export CUDA_VISIBLE_DEVICES=${gpu[$i]}
echo "use gpu id is ${gpu[$i]}, free memory is ${free_mem}"
python hello_world.py
