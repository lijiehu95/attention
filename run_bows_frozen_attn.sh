

source activate xai
export PYTHONPATH=/home/yila22/prj
export CUDA_VISIBLE_DEVICES='4'
exp_name="debug-freeze-att"


for dataset in sst imdb hate emotion;do
python train.py --dataset $dataset --data_dir . --output_dir test_outputs/ --attention frozen --encoder average  \
  --exp_name $exp_name
done;