


source activate xai
export PYTHONPATH=/home/yila22/prj
export CUDA_VISIBLE_DEVICES='8'
exp_name="debug-freeze-att-lstm"


#for dataset in sst imdb hate emotion;do
for dataset in hate emotion;do
python train.py --dataset $dataset --data_dir . --output_dir test_outputs/ --attention frozen --encoder lstm  \
  --exp_name $exp_name
done;

