

source activate xai
export PYTHONPATH=/home/yila22/prj
export CUDA_VISIBLE_DEVICES='5'
exp_name="freeze-att-average-v1"


#for dataset in sst imdb hate emotion;do
for dataset in hate offensive rotten_tomatoes;do
  com="python train.py --dataset $dataset --data_dir . --output_dir test_outputs/ --attention frozen --encoder average  \
    --exp_name $exp_name"
  nohup $com > $exp_name-$dataset.log 2>&1 &
done;