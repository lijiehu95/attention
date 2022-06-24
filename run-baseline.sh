source activate xai
export PYTHONPATH=${PYTHONPATH}:/home/yila22/prj:/mnt/yixin/
export CUDA_VISIBLE_DEVICES='0'
exp_name="baseline"
for model in simple-rnn lstm; do
for dataset in sst imdb ; do
  com="python train.py --dataset $dataset --data_dir . --output_dir test_outputs/ --attention tanh \
    --encoder $model --exp_name $exp_name"
  nohup $com > $exp_name-$model-$dataset-$RANDOM.log 2>&1 &
done;done;