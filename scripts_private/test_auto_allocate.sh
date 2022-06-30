## config
gpu=(2 3 4 5 6 7 8 9)
gpunum=${#gpu[@]}
task_load=8000

# find suitable gpu
id_idx=0
free_mem=0
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
gpu_id=${gpu[$i]}
free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $gpu_id | grep -Eo "[0-9]+")
export CUDA_VISIBLE_DEVICES=$gpu_id
echo "use gpu id is ${gpu[$i]}, free memory is ${free_mem}"

# your command here
python hello_world.py
