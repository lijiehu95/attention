source activate xai
export PYTHONPATH=/home/yila22/prj
export CUDA_VISIBLE_DEVICES='2,8'
exp_name="debug-baseline"
python trian.py --dataset ${1} --data_dir . --output_dir test_outputs/ --attention tanh --encoder lstm
