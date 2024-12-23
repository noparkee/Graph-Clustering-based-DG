import os
import sys
import argparse
os.environ["MKL_THREADING_LAYER"] = "GNU"

import torch

from src.config import GVRT_Config, GDG_Config, print_configs
from src.data import get_datasets_and_iterators
from src.algorithms import get_algorithm_class
from src.train import Trainer
from src.utils import Print, set_seeds, set_output

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', ''):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser('Train a Domain Generalization Model for the CUB-DG dataset')
parser.add_argument('--algorithm', help='Domain generalization algorithm')
parser.add_argument('--test-env', type=int, help='test environment (used for multi-source DG)')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
parser.add_argument('--output-log', help='output log dir name (default: stdout and without saving)')

parser.add_argument('--attn-layers', type=str, default='')

parser.add_argument('--use-vg', type=str2bool, default=True)
parser.add_argument('--use-tg', type=str2bool, default=True)

parser.add_argument('--global-align-loss', type=str2bool, default=True)
parser.add_argument('--graph-loss', type=str2bool, default=True)
parser.add_argument('--cluster-loss', type=str2bool, default=True)
parser.add_argument('--matching-loss', type=str2bool, default=True)
parser.add_argument('--matching-cls-loss', type=str2bool, default=True)
parser.add_argument('--global-align-loss-lambda', type=float, default=1.0)
parser.add_argument('--graph-loss-lambda', type=float, default=1.0)
parser.add_argument('--cluster-loss-lambda', type=float, default=1.0)
parser.add_argument('--matching-loss-lambda', type=float, default=0.1)
parser.add_argument('--matching-cls-loss-lambda', type=float, default=0.1)

parser.add_argument('--num-v-neighbors', type=int, default=8)
parser.add_argument('--num-t-neighbors', type=int, default=3)

parser.add_argument('--proj-dim', type=int, default=512)
parser.add_argument('--text-dim', type=int, default=512)
parser.add_argument('--graph-dim', type=int, default=2048)

parser.add_argument('--v-clusters', type=int, default=5)
parser.add_argument('--t-clusters', type=int, default=3)


def main():
    args = vars(parser.parse_args())
    gvrt_flag, gdg_flag = False, False
    if args["algorithm"] == "GVRT":
        gvrt_flag = True
        config = GVRT_Config(False)
    elif args["algorithm"] == "GDG":
        gdg_flag = True
        config = GDG_Config(args)
    env_flag = args["test_env"]
    output, writer, save_prefix = set_output(args, "train_model_log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_dict = config.__dict__ if args["algorithm"] == "GDG" else None
    set_seeds(args["algorithm"], env_flag, args["seed"])
    print_configs(args, device, output, config_dict)
    if config_dict is not None:
        print(config_dict)

    ## Loading datasets
    start = Print(" ".join(['start loading datasets']), output)
    datasets, iterators_train, iterators_eval, eval_names = get_datasets_and_iterators(env_flag, gvrt_flag, gdg_flag, args, eval_flag=False)
    end = Print('end loading datasets', output)
    Print(" ".join(['elapsed time:', str(end - start)]), output, newline=True)

    ## setup trainer configurations
    start = Print('start setting trainer configurations', output)
    algorithm_class = get_algorithm_class(args["algorithm"])
    if gvrt_flag or gdg_flag:
        model = algorithm_class(datasets[0].num_classes, datasets[0].vocab, config)
    else:
        model = algorithm_class(datasets[0].num_classes)
    trainer = Trainer(model, device, args)
    end = Print('end setting trainer configurations', output)
    Print(" ".join(['elapsed time:', str(end - start)]), output, newline=True)

    ## train a model
    BEST_SCORE = 0
    N_STEPS, CHECKPOINT_FREQ = 5000, 300
    start = Print('start training a model', output)
    trainer.headline("step", model.loss_names, eval_names, output)
    for step in range(N_STEPS):
        ### train
        minibatches = next(iterators_train)
        trainer.train(minibatches)
        
        if (step + 1) % 10 == 0:
            print('# step [{}/{}]'.format(step + 1, N_STEPS), end='\r', file=sys.stderr)
        if step == 0 or ((step + 1) % CHECKPOINT_FREQ == 0) or (step + 1 == N_STEPS):
            for iterator_eval, eval_name in zip(iterators_eval, eval_names):
                for B, minibatch in enumerate(iterator_eval):
                    trainer.evaluate(minibatch, eval_name)
                    if B % 5 == 0: print('# step [{}/{}] {} {:.1%}'.format(step + 1, N_STEPS, eval_name, B / len(iterator_eval)), end='\r', file=sys.stderr)
                print(' ' * 50, end='\r', file=sys.stderr)

            avg_val = trainer.log(step + 1, output, writer, test_env=args["test_env"])
            if avg_val > BEST_SCORE:
                trainer.save_model(step+1, save_prefix, best_flag=True)
                BEST_SCORE = avg_val
            elif avg_val == BEST_SCORE:
                trainer.save_model(step+1, save_prefix, best_flag=False, same_flag=True)
                BEST_SCORE = avg_val
            else:
                trainer.save_model(step+1, save_prefix, best_flag=False)
                
            if step + 1 == N_STEPS:
                trainer.save_model(step + 1, save_prefix)

    end = Print('end training a model', output)
    Print(" ".join(['elapsed time:', str(end - start)]), output, newline=True)
    if not output == sys.stdout: output.close()


if __name__ == '__main__':
    main()
