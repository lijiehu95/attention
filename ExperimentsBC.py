from attention.configurations import generate_config
from attention.Trainers.TrainerBC import Trainer, Evaluator
            
def train_dataset(dataset, args, config='lstm') :
        config = generate_config(dataset, args, config)
        trainer = Trainer(dataset, args, config=config)
        #go ahead and save model
        dirname = trainer.model.save_values(save_model=False)
        print("DIRECTORY:", dirname)
        if args.adversarial :
            trainer.train_adversarial(dataset.train_data, dataset.test_data, args)
        elif args.ours:
            from attention.attack import PGDAttacker
            PGDer = PGDAttacker(
                radius=args.pgd_radius, steps=args.pgd_step, step_size=args.pgd_step_size, random_start= \
                True, norm_type=args.pgd_norm_type, ascending=True
            )
            trainer.PGDer = PGDer
            X_PGDer = PGDAttacker(
                radius=args.x_pgd_radius, steps=args.x_pgd_step, step_size=args.x_pgd_step_size, random_start= \
                    True, norm_type=args.x_pgd_norm_type, ascending=True
            )
            trainer.PGDer = PGDer
            trainer.X_PGDer = X_PGDer
            trainer.train_ours(dataset.train_data, dataset.test_data, args)
        else:
            trainer.train_standard(dataset.train_data, dataset.test_data, args, save_on_metric=dataset.save_on_metric)
        print('####################################')
        print("TEST RESULTS FROM BEST MODEL")
        evaluator = Evaluator(dataset, trainer.model.dirname, args)
        _ = evaluator.evaluate(dataset.test_data, save_results=True)
        return trainer, evaluator

def train_dataset_on_encoders(dataset, args, exp_name) :
	train_dataset(dataset, args, exp_name)
