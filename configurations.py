import os
import git

lambda_config = {'emoji': {
    'simple-rnn': {'dataset': 'emoji', 'encoder': 'simple-rnn', 'att jsd decrease ratio': 0.00935360713704422,
                   'tvd_decrease ratio': 0.2897150917629588, 'lambda_1': 1.0, 'lambda_2': 1.0, 'x_pgd_radius': 0.01,
                   'pgd_radius': 0.02, 'final_metric.micro avg/f1-score': 0.8276678829780424,
                   'original_metric.micro avg/f1-score': 0.828628128800973,
                   'baseline_px_jsd_att_diff_te': 0.003430581416606342,
                   'baseline_px_tvd_pred_diff_te': 2.6801102859828454, 'our_px_jsd_att_diff': 0.0033984931057837622,
                   'our_px_tvd_pred_diff': 1.903641888544476, 'score': 0.299068698900003},
    'lstm': {'dataset': 'emoji', 'encoder': 'lstm', 'att jsd decrease ratio': 0.829323395102648,
             'tvd_decrease ratio': 0.4017541520819078, 'lambda_1': 0.0, 'lambda_2': 0.0001, 'x_pgd_radius': 0.01,
             'pgd_radius': 0.02, 'final_metric.micro avg/f1-score': 0.8270917354842839,
             'original_metric.micro avg/f1-score': 0.8277318993662378,
             'baseline_px_jsd_att_diff_te': 0.003973516641782317, 'baseline_px_tvd_pred_diff_te': 3.553233247759949,
             'our_px_jsd_att_diff': 0.0006781863299225337, 'our_px_tvd_pred_diff': 2.125707037156908,
             'score': 1.2310775471845559}}, 'hate': {
    'simple-rnn': {'dataset': 'hate', 'encoder': 'simple-rnn', 'att jsd decrease ratio': 0.7042683604031155,
                   'tvd_decrease ratio': 0.35741259581089146, 'lambda_1': 1.0, 'lambda_2': 0.0, 'x_pgd_radius': 0.01,
                   'pgd_radius': 0.02, 'final_metric.micro avg/f1-score': 0.5072463768115942,
                   'original_metric.micro avg/f1-score': 0.4974721941354904,
                   'baseline_px_jsd_att_diff_te': 0.0006748235746367285,
                   'baseline_px_tvd_pred_diff_te': 6.009005337642185, 'our_px_jsd_att_diff': 0.0001995666821659502,
                   'our_px_tvd_pred_diff': 3.8613111416739887, 'score': 1.061680956214007},
    'lstm': {'dataset': 'hate', 'encoder': 'lstm', 'att jsd decrease ratio': 0.40684999740724265,
             'tvd_decrease ratio': 0.4236371338357911, 'lambda_1': 1.0, 'lambda_2': 0.0, 'x_pgd_radius': 0.01,
             'pgd_radius': 0.02, 'final_metric.micro avg/f1-score': 0.5160094371418942,
             'original_metric.micro avg/f1-score': 0.5183687226154364,
             'baseline_px_jsd_att_diff_te': 0.0011689751958479813, 'baseline_px_tvd_pred_diff_te': 5.417680773463442,
             'our_px_jsd_att_diff': 0.0006933776404480992, 'our_px_tvd_pred_diff': 3.122550018556117,
             'score': 0.8304871312430337}}, 'imdb': {
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
             'score': 0.8578301882225376}}, 'rotten_tomatoes': {
    'simple-rnn': {'dataset': 'rotten_tomatoes', 'encoder': 'simple-rnn', 'att jsd decrease ratio': 0.28787142840295393,
                   'tvd_decrease ratio': 0.0717429018824509, 'lambda_1': 1.0, 'lambda_2': 0.0001, 'x_pgd_radius': 0.01,
                   'pgd_radius': 0.01, 'final_metric.micro avg/f1-score': 0.7811320754716982,
                   'original_metric.micro avg/f1-score': 0.7858490566037736,
                   'baseline_px_jsd_att_diff_te': 0.00032027948226006524,
                   'baseline_px_tvd_pred_diff_te': 8.001865225468041, 'our_px_jsd_att_diff': 0.00022808017021370167,
                   'our_px_tvd_pred_diff': 7.4277881937206915, 'score': 0.35961433028540485},
    'lstm': {'dataset': 'rotten_tomatoes', 'encoder': 'lstm', 'att jsd decrease ratio': 0.2843444937881119,
             'tvd_decrease ratio': 0.20751588802558604, 'lambda_1': 1.0, 'lambda_2': 0.0001, 'x_pgd_radius': 0.01,
             'pgd_radius': 0.01, 'final_metric.micro avg/f1-score': 0.7915094339622641,
             'original_metric.micro avg/f1-score': 0.7915094339622641,
             'baseline_px_jsd_att_diff_te': 0.00048524034615904805, 'baseline_px_tvd_pred_diff_te': 6.916444557117966,
             'our_px_jsd_att_diff': 0.00034726492556488533, 'our_px_tvd_pred_diff': 5.4811724228679015,
             'score': 0.49186038181369796}}, 'sentiment': {
    'simple-rnn': {'dataset': 'sentiment', 'encoder': 'simple-rnn', 'att jsd decrease ratio': 0.2776055346416866,
                   'tvd_decrease ratio': 0.3689766763296379, 'lambda_1': 0.0001, 'lambda_2': 0.0001,
                   'x_pgd_radius': 0.01, 'pgd_radius': 0.01, 'final_metric.micro avg/f1-score': 0.6785497879216321,
                   'original_metric.micro avg/f1-score': 0.6579478893152898,
                   'baseline_px_jsd_att_diff_te': 0.0005040193971114946,
                   'baseline_px_tvd_pred_diff_te': 3.7021035469317534, 'our_px_jsd_att_diff': 0.00036410082290657764,
                   'our_px_tvd_pred_diff': 2.336113684756711, 'score': 0.6465822109713244},
    'lstm': {'dataset': 'sentiment', 'encoder': 'lstm', 'att jsd decrease ratio': 0.3115398162523742,
             'tvd_decrease ratio': 0.2839767259117719, 'lambda_1': 0.0001, 'lambda_2': 0.0001, 'x_pgd_radius': 0.01,
             'pgd_radius': 0.01, 'final_metric.micro avg/f1-score': 0.6818824479903052,
             'original_metric.micro avg/f1-score': 0.6724904059785902,
             'baseline_px_jsd_att_diff_te': 0.0009300046929014273, 'baseline_px_tvd_pred_diff_te': 3.3693831757665085,
             'our_px_jsd_att_diff': 0.0006402712017610709, 'our_px_tvd_pred_diff': 2.4125567731701274,
             'score': 0.5955165421641462}}, 'sst': {
    'simple-rnn': {'dataset': 'sst', 'encoder': 'simple-rnn', 'att jsd decrease ratio': 0.15580707494856694,
                   'tvd_decrease ratio': 0.3580788756883194, 'lambda_1': 0.0001, 'lambda_2': 0.0, 'x_pgd_radius': 0.01,
                   'pgd_radius': 0.005, 'final_metric.micro avg/f1-score': 0.6633663366336634,
                   'original_metric.micro avg/f1-score': 0.6622662266226622,
                   'baseline_px_jsd_att_diff_te': 0.00033225718279026207,
                   'baseline_px_tvd_pred_diff_te': 0.7191493175473019, 'our_px_jsd_att_diff': 0.00028048916300906,
                   'our_px_tvd_pred_diff': 0.4616371384679418, 'score': 0.5138859506368864},
    'lstm': {'dataset': 'sst', 'encoder': 'lstm', 'att jsd decrease ratio': 0.4211355947975625,
             'tvd_decrease ratio': 0.3999350661382163, 'lambda_1': 0.0001, 'lambda_2': 0.0, 'x_pgd_radius': 0.01,
             'pgd_radius': 0.01, 'final_metric.micro avg/f1-score': 0.6094609460946094,
             'original_metric.micro avg/f1-score': 0.5786578657865786,
             'baseline_px_jsd_att_diff_te': 0.0007877381008061528, 'baseline_px_tvd_pred_diff_te': 4.13886646206754,
             'our_px_jsd_att_diff': 0.0004559935471784514, 'our_px_tvd_pred_diff': 2.483588629823313,
             'score': 0.8210706609357787}}, 'stance_abortion': {
    'simple-rnn': {'dataset': 'stance_abortion', 'encoder': 'simple-rnn', 'att jsd decrease ratio': 0.9210619477400288,
                   'tvd_decrease ratio': 0.9999800914257908, 'lambda_1': 1.0, 'lambda_2': 0.0001, 'x_pgd_radius': 0.01,
                   'pgd_radius': 0.02, 'final_metric.micro avg/f1-score': 0.8095238095238095,
                   'original_metric.micro avg/f1-score': 0.7445887445887447,
                   'baseline_px_jsd_att_diff_te': 0.0007173442899136606,
                   'baseline_px_tvd_pred_diff_te': 9.815632450632203, 'our_px_jsd_att_diff': 5.662576104559657e-05,
                   'our_px_tvd_pred_diff': 0.0001954152470543271, 'score': 1.9210420391658196},
    'lstm': {'dataset': 'stance_abortion', 'encoder': 'lstm', 'att jsd decrease ratio': 0.932171453100242,
             'tvd_decrease ratio': 0.9993427488595018, 'lambda_1': 1.0, 'lambda_2': 0.0, 'x_pgd_radius': 0.01,
             'pgd_radius': 0.02, 'final_metric.micro avg/f1-score': 0.8095238095238095,
             'original_metric.micro avg/f1-score': 0.7575757575757576,
             'baseline_px_jsd_att_diff_te': 0.0008734637230470398, 'baseline_px_tvd_pred_diff_te': 6.8386488512996975,
             'our_px_jsd_att_diff': 5.9245775103933395e-05, 'our_px_tvd_pred_diff': 0.004494709756983307,
             'score': 1.9315142019597438}}, 'stance_atheism': {
    'simple-rnn': {'dataset': 'stance_atheism', 'encoder': 'simple-rnn', 'att jsd decrease ratio': 0.09881768640303036,
                   'tvd_decrease ratio': 0.9998405710970648, 'lambda_1': 1.0, 'lambda_2': 1.0, 'x_pgd_radius': 0.01,
                   'pgd_radius': 0.02, 'final_metric.micro avg/f1-score': 0.5769230769230769,
                   'original_metric.micro avg/f1-score': 0.8021978021978022,
                   'baseline_px_jsd_att_diff_te': 5.904871328052e-05,
                   'baseline_px_tvd_pred_diff_te': 5.7351623603275845, 'our_px_jsd_att_diff': 5.321365604906313e-05,
                   'our_px_tvd_pred_diff': 0.0009143506432627584, 'score': 1.0986582575000952},
    'lstm': {'dataset': 'stance_atheism', 'encoder': 'lstm', 'att jsd decrease ratio': 0.561377408958236,
             'tvd_decrease ratio': 0.7779518896443083, 'lambda_1': 0.0, 'lambda_2': 0.0, 'x_pgd_radius': 0.01,
             'pgd_radius': 0.005, 'final_metric.micro avg/f1-score': 0.8461538461538461,
             'original_metric.micro avg/f1-score': 0.7967032967032966,
             'baseline_px_jsd_att_diff_te': 0.0002639804362123678, 'baseline_px_tvd_pred_diff_te': 8.361234256199428,
             'our_px_jsd_att_diff': 0.0001157877829158039, 'our_px_tvd_pred_diff': 1.8565962668303604,
             'score': 1.3393292986025442}}, 'stance_climate': {
    'simple-rnn': {'dataset': 'stance_climate', 'encoder': 'simple-rnn', 'att jsd decrease ratio': 0.030614141108817713,
                   'tvd_decrease ratio': 0.5067418436116862, 'lambda_1': 1.0, 'lambda_2': 1.0, 'x_pgd_radius': 0.01,
                   'pgd_radius': 0.02, 'final_metric.micro avg/f1-score': 0.7272727272727273,
                   'original_metric.micro avg/f1-score': 0.7045454545454546,
                   'baseline_px_jsd_att_diff_te': 7.191543806005608e-05,
                   'baseline_px_tvd_pred_diff_te': 0.1722239012067968, 'our_px_jsd_att_diff': 6.971380869138308e-05,
                   'our_px_tvd_pred_diff': 0.0849508439952677, 'score': 0.5373559847205038},
    'lstm': {'dataset': 'stance_climate', 'encoder': 'lstm', 'att jsd decrease ratio': 0.4659328952681835,
             'tvd_decrease ratio': 0.7594695040693601, 'lambda_1': 0.0001, 'lambda_2': 0.0, 'x_pgd_radius': 0.01,
             'pgd_radius': 0.02, 'final_metric.micro avg/f1-score': 0.75,
             'original_metric.micro avg/f1-score': 0.6818181818181818,
             'baseline_px_jsd_att_diff_te': 0.000236697190054904, 'baseline_px_tvd_pred_diff_te': 6.453930334611372,
             'our_px_jsd_att_diff': 0.00012641218299077911, 'our_px_tvd_pred_diff': 1.5523670640858738,
             'score': 1.2254023993375436}}, 'stance_feminist': {
    'simple-rnn': {'dataset': 'stance_feminist', 'encoder': 'simple-rnn', 'att jsd decrease ratio': 0.395617466242285,
                   'tvd_decrease ratio': 0.24501871684526985, 'lambda_1': 1.0, 'lambda_2': 0.0001, 'x_pgd_radius': 0.01,
                   'pgd_radius': 0.02, 'final_metric.micro avg/f1-score': 0.7244444444444443,
                   'original_metric.micro avg/f1-score': 0.7111111111111111,
                   'baseline_px_jsd_att_diff_te': 0.0010431584114364038,
                   'baseline_px_tvd_pred_diff_te': 2.771755006776916, 'our_px_jsd_att_diff': 0.0006304667238146068,
                   'our_px_tvd_pred_diff': 2.0926231516069835, 'score': 0.6406361830875549},
    'lstm': {'dataset': 'stance_feminist', 'encoder': 'lstm', 'att jsd decrease ratio': 0.779406867530635,
             'tvd_decrease ratio': 0.34984538722634545, 'lambda_1': 1.0, 'lambda_2': 0.0, 'x_pgd_radius': 0.01,
             'pgd_radius': 0.005, 'final_metric.micro avg/f1-score': 0.7866666666666666,
             'original_metric.micro avg/f1-score': 0.7955555555555557,
             'baseline_px_jsd_att_diff_te': 0.0004273226360479991, 'baseline_px_tvd_pred_diff_te': 5.453808544079464,
             'our_px_jsd_att_diff': 9.426443886089449e-05, 'our_px_tvd_pred_diff': 3.545818782117632,
             'score': 1.1292522547569805}}, 'stance_hillary': {
    'simple-rnn': {'dataset': 'stance_hillary', 'encoder': 'simple-rnn', 'att jsd decrease ratio': 0.03779696987295346,
                   'tvd_decrease ratio': 0.4827540204039457, 'lambda_1': 0.0001, 'lambda_2': 0.0001,
                   'x_pgd_radius': 0.01, 'pgd_radius': 0.01, 'final_metric.micro avg/f1-score': 0.7338709677419355,
                   'original_metric.micro avg/f1-score': 0.7177419354838711,
                   'baseline_px_jsd_att_diff_te': 6.337665012500609e-05,
                   'baseline_px_tvd_pred_diff_te': 0.6281706479287916, 'our_px_jsd_att_diff': 6.098120478958252e-05,
                   'our_px_tvd_pred_diff': 0.324918742141416, 'score': 0.5205509902768992},
    'lstm': {'dataset': 'stance_hillary', 'encoder': 'lstm', 'att jsd decrease ratio': 0.7919451719429675,
             'tvd_decrease ratio': 0.2178594518231827, 'lambda_1': 0.0001, 'lambda_2': 0.0, 'x_pgd_radius': 0.01,
             'pgd_radius': 0.01, 'final_metric.micro avg/f1-score': 0.7056451612903226,
             'original_metric.micro avg/f1-score': 0.7258064516129032,
             'baseline_px_jsd_att_diff_te': 0.0006985496288360728, 'baseline_px_tvd_pred_diff_te': 8.29016774700534,
             'our_px_jsd_att_diff': 0.00014533662291679291, 'our_px_tvd_pred_diff': 6.484076346120527,
             'score': 1.0098046237661502}}}


def generate_config(dataset, args, exp_name):
    repo = git.Repo(search_parent_directories=True)

    if args.encoder == 'lstm':
        enc_type = 'rnn'
    elif args.encoder == 'average':
        enc_type = args.encoder
    elif args.encoder == 'simple-rnn':
        enc_type = args.encoder
    else:
        raise Exception("unknown encoder type")

    config = {
        'model': {
            'encoder': {
                'vocab_size': dataset.vec.vocab_size,
                'embed_size': dataset.vec.word_dim,
                'type': enc_type,
                'hidden_size': args.hidden_size
            },
            'decoder': {
                'attention': {
                    'type': 'tanh'
                },
                'output_size': dataset.output_size
            }
        },
        'training': {
            'bsize': dataset.bsize if hasattr(dataset, 'bsize') else 32,
            'weight_decay': 1e-5,
            'pos_weight': dataset.pos_weight if hasattr(dataset, 'pos_weight') else None,
            'basepath': dataset.basepath if hasattr(dataset, 'basepath') else 'outputs',
            'exp_dirname': os.path.join(dataset.name, exp_name)
        },
        'git_info': {
            'branch': repo.active_branch.name,
            'sha': repo.head.object.hexsha
        },
        'command': args.command
    }

    if args.encoder == 'average':
        config['model']['encoder'].update({'projection': True, 'activation': 'tanh'})

    return config
