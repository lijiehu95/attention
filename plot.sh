for dataset in hate rotten_tomatoes  imdb sst emoji;do
for att in tanh;do
for model_type in simple-rnn; do
python seed_graphs.py --dataset $dataset --model_type $model_type+$att
done; done; done;
