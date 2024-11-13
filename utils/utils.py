import logging
from utils.EarlyStop import EarlyStoppingCriterion
import torch
import numpy as np
import random
from tqdm import tqdm
import dgl
from models.models import DAGCL
import os

def choose_model(args, dataloader):
    if args.model == 'DAGCL':
        return DAGCL(args, dataloader)

def construct_negative_graph(graph, etype):
    utype, _ , vtype = etype
    src, _ = graph.edges(etype = etype)
    dst = torch.randint(graph.num_nodes(vtype), size = src.shape, device = src.device)
    return dgl.heterograph({etype: (src, dst)}, num_nodes_dict = {ntype: graph.number_of_nodes(ntype) for ntype in graph.ntypes}).to(graph.device)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def config(args):
    setup_seed(args.seed)

    path = f"{args.dataset}_model_{args.model}_lr_{args.lr}_embed_size_{args.embed_size}_batch_size_{args.batch_size}_weight_decay_{args.weight_decay}_layers_{args.layers}_neg_number_{args.neg_number}_seed_{args.seed}_k_{args.k}_sigma_{args.sigma}_gamma_{args.gamma}_beta_class_{args.beta_class}_degree_threshold_{args.degree_threshold}"
    if os.path.exists('./logs/' + path + '.log'):
        os.remove('./logs/' + path + '.log')

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s  %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename='./logs/' + path + '.log')
    logger = logging.getLogger()
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)
    early_stop = EarlyStoppingCriterion(patience = args.patience, save_path = './logs/' + path + '.pt')
    return early_stop

