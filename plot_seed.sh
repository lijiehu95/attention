for dataset in hate rotten_tomatoes  imdb sst emoji  \
                sentiment  stance_abortion  stance_atheism  stance_climate  stance_feminist  \
                stance_hillary;do
for att in ours;do
for model_type in lstm simple-rnn; do
python seed_graphs.py --dataset $dataset --model_type $model_type+$att
done; done; done;