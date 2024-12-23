import os
import sys
import random
import hashlib
import datetime
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
import dgl


def Print(string, output, newline=False, timestamp=True):
    """ print to stdout and a file (if given) """
    if timestamp:
        time = datetime.datetime.now()
        line = '\t'.join([str(time.strftime('%m-%d %H:%M:%S')), string])
    else:
        time = None
        line = string

    if not output == sys.stdout:
        print(line, file=output)
        if newline: print("", file=output)
    else:
        print(line, file=sys.stderr)
        if newline: print("", file=sys.stderr)

    output.flush()
    return time


def seed_hash(*args):
    """ derive an integer hash from all args, for use as a random seed """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)


def set_seeds(algorithm, test_env, seed):
    """ set random seeds """
    seed = seed_hash(algorithm, test_env, seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms(mode=True)


def set_output(args, string):
    """ set output configurations """
    output, save_prefix, writer = sys.stdout, None, None
    if args["output_log"] is not None:
        log_name = args["output_log"]
        save_prefix = os.path.join("results", os.path.join(log_name, "testenv" + str(args["test_env"])))
        if not os.path.exists(save_prefix):
            os.makedirs(save_prefix, exist_ok=True)
        output = open(os.path.join(save_prefix, "%s.txt" % string), "a")
        
        if "eval" not in string:        # tensorboard
            tb = os.path.join(save_prefix, "tensorboard")
            if not os.path.exists(tb):
                os.makedirs(tb, exist_ok=True)
            writer = SummaryWriter(tb)

    return output, writer, save_prefix
