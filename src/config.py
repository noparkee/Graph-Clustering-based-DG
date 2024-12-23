import os
import sys
import json

import torch

from src.utils import Print


class GVRT_Config():
    def __init__(self, ste_flag):
        """ Grounding Visual Representations with Texts (GVRT) algorithm configurations """
        self.align_loss = True
        self.expl_loss = True
        self.align_loss_lambda = 1.0
        self.expl_loss_lambda = 1.0
        self.ste_flag = ste_flag
        self.proj_size = 128
        self.lstm_size = 128
        self.embed_size = 512 if not self.ste_flag else self.lstm_size 


class GDG_Config():
    def __init__(self, args):
        self.attn_layers = args["attn_layers"]
        self.use_vg = args["use_vg"]
        self.use_tg = args["use_tg"]
        self.global_align_loss = args["global_align_loss"]
        self.graph_loss = args["graph_loss"]
        self.cluster_loss = args["cluster_loss"]
        self.matching_loss = args["matching_loss"]
        self.matching_cls_loss = args["matching_cls_loss"]
        self.global_align_loss_lambda = args["global_align_loss_lambda"]
        self.graph_loss_lambda = args["graph_loss_lambda"]
        self.cluster_loss_lambda = args["cluster_loss_lambda"]
        self.matching_loss_lambda = args["matching_loss_lambda"]
        self.matching_cls_loss_lambda = args["matching_cls_loss_lambda"]
        self.num_v_neighbors = args["num_v_neighbors"]
        self.num_t_neighbors = args["num_t_neighbors"]
        self.proj_dim = args["proj_dim"]
        self.text_dim = args["text_dim"]
        self.graph_dim = args["graph_dim"]
        self.v_clusters = args["v_clusters"]
        self.t_clusters = args["t_clusters"]


class SWAD_Config():
    def __init__(self, args, domainbed=False):        
        if domainbed:
            self.n_converge = args.swad_n_converge
            self.n_tolerance = args.swad_n_tolerance
            self.tolerance_ratio = args.swad_tolerance_ratio
        else:
            self.n_converge = args['swad_n_converge']
            self.n_tolerance = args['swad_n_tolerance']
            self.tolerance_ratio = args['swad_tolerance_ratio']
                    

def print_configs(args, device, output, config_dict=None):
    Print(" ".join(['##### arguments #####']), output)
    Print(" ".join(['algorithm:', str(args["algorithm"])]), output)
    if args["algorithm"] == "GVRT":
        Print(" ".join(['ste:', str(False)]), output)
    if args["algorithm"] == "GDG":
        keys, values = config_dict.keys(), config_dict.values()
        for (k, v) in zip(keys, values):
            Print("\t" + " ".join([str(k) + ":", str(v)]), output)
    Print(" ".join(['test_env:', str(args["test_env"])]), output)
    Print(" ".join(['seed:', str(args["seed"])]), output)
    if "checkpoint" in args:
        Print(" ".join(['checkpoint:', str(args["checkpoint"])]), output)
    Print(" ".join(['output_log:', str(args["output_log"])]), output)
    Print(" ".join(['device: %s (%d GPUs)' % (device, torch.cuda.device_count())]), output)
