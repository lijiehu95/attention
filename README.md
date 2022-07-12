# REAL Attention 

We've based our repository on the [code](https://github.com/sarahwie/attention) provided by Sarthak Jain & Byron Wallace for their paper *[Attention is not not Explanation](https://arxiv.org/abs/1908.04626)*.

# Envrioment Setup
1. Ensure you have installed python environment in your computer. Anaconda is recommended. We use Python 3.6 in our experiments.


2. Use the following command to install the dependencies:
```shell
pip install -r requirements.txt
python -m spacy download en
```

3. Preprocess the dataset using the following command. Note that our processing method is slighting different from the \
original implementation in the reference repository.
```shell
python prepare_data.py
```

5. Setup the enviroment variable using the following command:
```shell
your_path_to_the_repository="XXX" # Replace XXX with your path to the repository 
export PYTHONPATH="$PYTHONPATH:$your_path_to_this_repository"
```

6. Activate your conda env and now you are ready to run the experiments.
```shell
conda activate XXX # Replace XXX with your conda env name
```
6. **Wandb Logging**: 


Since we use wandb to log experiments, you are required to register a wandb account , activate that on your computer and revise the argument ``args.wandb_entity``. You can register your account at [wandb.com](https://wandb.com/signup) and read about the quick start guide at [wandb.com](https://docs.wandb.ai/v/zh-hans/quickstart).

# Run Our Main Experiments
## 1. Baseline model
First train the baseline models and then the model ckpt and attention score will be saved in the output directory
```shell
bash run-baseline.sh
```

## 2. Our model
Then train our model based on the baseline model using the following command. Note that the performance of the baseline model will be computed at the start of training.
```shell
bash run-adversarial_ours.sh
```

# Tasks breakdown for adding bert encoder 
To add bert encoder based on this code, you might need to do the following:
1. Align the bert data tokenize format, which you need to add ``[SEQ]`` in the front and end of the text to match existing code.
2. Add Bert Dataset loading Module.
3. Add Bert Encoder Module.

# The repository is organized as follows:
1. ``train.py`` as the main training python module.
2. ``preprocess.py`` as the preprocessing dataset module for encoder that use word-embedding.
3. ``attack.py`` as the adversarial pgd attack module.

# Run Additional Experiments based on the reference paper

## Running Random Seeds Experiments 
```shell
bash run_seeds.sh
```
- Code for constructing the violin plots can be found in `seed_graphs.py` and `Seed_graphs.ipynb`.

## Freezing the Attention Distribution
```shell
bash run_frozen_attention.sh
```

## Running BOWs Experiments 
- To run the Bag of Words model with trained (MLP) attention weights: `./run_bows_baselines.sh [DATASET]`
- To run the Bag of Words model with uniform attention weights: `./run_bows_frozen_attn.sh [DATASET]`
- To run the Bag of Words model with frozen attention weights from another model: `./run_bows_set_to_pretrained_distribution.sh [DATASET] [path/to/saved/model/with/attention/weights]`
