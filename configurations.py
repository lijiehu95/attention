import os
import git

lambda_config = {
    'simple-rnn': {'dataset': 'imdb', 'encoder': 'simple-rnn', 'att jsd decrease ratio': 0.2111088406538017,
                   'tvd_decrease ratio': 0.3985042182822998, 'lambda_1': 1.0, 'lambda_2': 0.0001, 'x_pgd_radius': 0.01,
                   'pgd_radius': 0.005, 'final_metric.micro avg/f1-score': 0.8779502360188816,
                   'original_metric.micro avg/f1-score': 0.8978718297463797,
                   'baseline_px_jsd_att_diff_te': 2.1446077921420805e-05,
                   'baseline_px_tvd_pred_diff_te': 7.6107025171854135, 'our_px_jsd_att_diff': 1.6918621274858568e-05,
                   'our_px_tvd_pred_diff': 4.577805459995308, 'score': 0.6096130589361015},
    'lstm': {'dataset': 'imdb', 'encoder': 'lstm', 'att jsd decrease ratio': 0.4246620643073442,
             'tvd_decrease ratio': 0.43316812391519344, 'lambda_1': 0.0001, 'lambda_2': 0.0, 'x_pgd_radius': 0.01,
             'pgd_radius': 0.02, 'final_metric.micro avg/f1-score': 0.8969917593407474,
             'original_metric.micro avg/f1-score': 0.8859108728698296,
             'baseline_px_jsd_att_diff_te': 2.372707687970209e-05, 'baseline_px_tvd_pred_diff_te': 8.488552659156275,
             'our_px_jsd_att_diff': 1.3651087431988741e-05, 'our_px_tvd_pred_diff': 4.811582229034225,
             'score': 0.8578301882225376}}


def generate_config(dataset, args, exp_name) :
    
    repo = git.Repo(search_parent_directories=True)

    if args.encoder == 'lstm' :
        enc_type = 'rnn'
    elif args.encoder == 'average' :
        enc_type = args.encoder
    elif args.encoder == 'simple-rnn' :
        enc_type = args.encoder
    else :
        raise Exception("unknown encoder type")

    config = {
        'model' :{
            'encoder' : {
                'vocab_size' : dataset.vec.vocab_size,
                'embed_size' : dataset.vec.word_dim,
		'type' : enc_type,
		'hidden_size' : args.hidden_size
            },
            'decoder' : {
                'attention' : {
                    'type' : 'tanh'
                },
                'output_size' : dataset.output_size
            }
        },
        'training' : {
            'bsize' : dataset.bsize if hasattr(dataset, 'bsize') else 32,
            'weight_decay' : 1e-5,
            'pos_weight' : dataset.pos_weight if hasattr(dataset, 'pos_weight') else None,
            'basepath' : dataset.basepath if hasattr(dataset, 'basepath') else 'outputs',
            'exp_dirname' : os.path.join(dataset.name, exp_name)
        },
        'git_info' : {
            'branch' : repo.active_branch.name,
            'sha' : repo.head.object.hexsha
        },
        'command' : args.command
    }

    if args.encoder == 'average' :
    	config['model']['encoder'].update({'projection' : True, 'activation' : 'tanh'})

    return config

