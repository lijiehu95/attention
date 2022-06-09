export CUDA_VISIBLE_DEVICES=2
source activate xai
export PYTHONPATH=/home/yila22/prj
exp_name="debug-and-find-lambda-v1"

for data in sst ; do
  for lambda_1 in  1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7;do
    for lambda_2 in  1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7;do
python train_and_run_experiments_bc.py --dataset $data \
  --data_dir . --output_dir test_adv_outputs/ \
  --encoder lstm --ours --lambda_1 $lambda_1 --lambda_2 $lambda_2 \
  --gold_label_dir /home/yila22/prj/attention/test_outputs/sst/lstm+tanh/Mon_Jun__6_13:13:30_2022/ --n_iters 80
done;done;done

for data in imdb ; do
  for lambda_1 in  1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7;do
    for lambda_2 in  1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7;do
python train_and_run_experiments_bc.py --dataset $data \
  --data_dir . --output_dir test_adv_outputs/ \
  --encoder lstm --ours --lambda_1 $lambda_1 --lambda_2 $lambda_2 \
  --gold_label_dir /home/yila22/prj/attention/test_outputs/imdb/lstm+tanh/Thu_Jun__9_07:14:57_2022 --n_iters 80
done;done;done
