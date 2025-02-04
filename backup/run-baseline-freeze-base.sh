source activate xai
export PYTHONPATH=/home/yila22/prj
export CUDA_VISIBLE_DEVICES='9'
exp_name="baseline-final"

n_iters=80
#datasets=("offensive" "irony" "rotten_tomatoes")
# imdb sst hate offensive rotten_tomatoes irony
for dataset in hate  ; do
  com="python train.py --dataset $dataset --data_dir . --output_dir test_outputs/ --attention tanh --encoder lstm --exp_name $exp_name --n_iters $n_iters"
#  $com
  nohup $com > $exp_name-$dataset.log 2>&1 &
done;
export CUDA_VISIBLE_DEVICES='4'
exp_name="freeze-att-lstm-final"
for dataset in  hate;do
  com="python train.py --dataset $dataset --data_dir . --output_dir test_outputs/ --attention frozen --encoder lstm  \
    --exp_name $exp_name --n_iters $n_iters"
#  $com
  nohup $com > $exp_name-$dataset.log 2>&1 &
done;

