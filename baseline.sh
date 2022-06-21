source activate xai
export PYTHONPATH=/home/yila22/prj
export CUDA_VISIBLE_DEVICES='2,8'
exp_name="debug-baseline"
python train.py --dataset sst --data_dir . --output_dir test_outputs/ --attention tanh --encoder lstm --exp_name
python train.py --dataset imdb --data_dir . --output_dir test_outputs/ --attention tanh --encoder lstm --exp_name
python train.py --dataset hate --data_dir . --output_dir test_outputs/ --attention tanh --encoder lstm --exp_name
python train.py --dataset emotion --data_dir . --output_dir test_outputs/ --attention tanh --encoder lstm --exp_name
