source activate xai
export PYTHONPATH=/home/yila22/prj
export CUDA_VISIBLE_DEVICES='9'
exp_name="baseline-v1"
for dataset in sst imdb hate rotten_tomatoes; do
python train.py --dataset $dataset --data_dir . --output_dir test_outputs/ --attention tanh \
  --encoder simple-rnn --exp_name $exp_name
done;