source activate xai
export PYTHONPATH=${PYTHONPATH}:/home/yila22/prj:/mnt/yixin/
export CUDA_VISIBLE_DEVICES='0'
exp_name="baseline"
gpu=(2 3 4 5 6 7 8 9) # gpu list
gpunum=8 # gpu num
i=0 # gpu pointer


for model in simple-rnn lstm; do
for dataset in emoji sentiment stance_abortion stance_atheism stance_climate stance_feminist \
                stance_hillary ; do
  i=`expr $i % $gpunum`
   export CUDA_VISIBLE_DEVICES=${gpu[$i]}
   echo "use gpu id is ${gpu[$i]}"

   # your command here

  com="python train.py --dataset $dataset --data_dir . --output_dir test_outputs/ --attention tanh \
    --encoder $model --exp_name $exp_name --baseline"
  nohup $com > $exp_name-$model-$dataset-$RANDOM.log 2>&1 &
#$com
   i=`expr $i + 1`

done;done;
# put the following code in the for loop
# For xxxx

# Done;