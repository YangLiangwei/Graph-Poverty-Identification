import logging
from utils.EarlyStop import EarlyStoppingCriterion
import torch
import numpy as np
import random
from tqdm import tqdm
import dgl
from models.models import GCNModel, GATModel, SGCModel, APPNPModel, DGCN
import os
from utils.Dataloader import VillageDataset,AllDataset
from PIL import Image
import pdb
import clip
from dgl.nn import LabelPropagation
import networkx as nx
import walker
from gensim.models import Word2Vec

def choose_model(args, input_features, num_classes):
    if args.model == 'gcn':
        return GCNModel(args, input_features, num_classes)
    if args.model == 'gat':
        return GATModel(args, input_features, num_classes)
    if args.model == 'sgc':
        return SGCModel(args, input_features, num_classes)
    if args.model == 'appnp':
        return APPNPModel(args, input_features, num_classes)
    if args.model == 'dgcn':
        return DGCN(args, input_features, num_classes)

def choose_dataset(args):
    if args.dataset == 'all':
        dataset = AllDataset(args)
    else:
        dataset = VillageDataset(args)
    return dataset

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.seed(seed)

def config(args):
    setup_seed(args.seed)

    path = f"{args.dataset}_model_{args.model}_lr_{args.lr}_weight_decay_{args.weight_decay}_layers_{args.layers}_distance_{args.village_distance}_train_percent_{args.train_percent}_alpha_{args.alpha}_h_init_{args.h_init}_neighbor_number_{args.neighbor_number}_walkers_{args.walkers}_walk_length_{args.walk_length}_window_size_{args.window_size}_epochs_{args.epochs}_alpha_{args.alpha}_reweight_{args.reweight}"
    if os.path.exists('./logs/' + path + '.log'):
        os.remove('./logs/' + path + '.log')

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s  %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename='./logs/' + path + '.log')
    logger = logging.getLogger()
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)
    early_stop = EarlyStoppingCriterion(patience = args.patience, save_path = './best_models/' + path + '.pt')
    return early_stop

def centrality2vec(args):
    dist = int(args.village_distance)
    dist_degree = np.load('dist_degree' + '_' + str(dist) + '.npy')
    dist_core = np.load('dist_core' + '_' + str(dist) + '.npy')

    arg_degree = np.argsort(dist_degree)
    arg_core = np.argsort(dist_core)

    G = nx.Graph()

    num_nodes = arg_degree.shape[0]
    for i in range(num_nodes):
        for j in range(args.neighbor_number):
            G.add_edge(i, arg_degree[i, j])
            G.add_edge(i, arg_core[i, j])

    print('random walk')
    X = walker.random_walks(G, n_walks = args.walkers, walk_len = args.walk_length, alpha = 0.1)
    X = X.astype(str)
    X = X.tolist()

    print('trianing word2vec model')
    model = Word2Vec(X, vector_size=args.h_init, window=args.window_size, min_count=0,
                             workers=4, epochs = args.epochs)

    emb_ls = []
    for i in range(num_nodes):
        node_emb = model.wv[str(i)]
        emb_ls.append(node_emb)
    return np.stack(emb_ls)

def get_clip_feature(args, model, preprocess, device):
    sheet_name = ['恩施市', '利川市', '建始县', '巴东县', '宣恩县', '咸丰县', '来凤县', '鹤峰县']

    ls = []
    for sheet in sheet_name:
        length = len(os.listdir('data/' + sheet + '/pic'))
        length = int(length / 12)

        for i in tqdm(range(length)):
            image = preprocess(Image.open('data/' + sheet + '/pic/' + str(i) + '_zoom_' + str(args.zoom) + '_' + args.type + '.png')).unsqueeze(0).to(device)
            ls.append(image)

        length = len(os.listdir('data/' + sheet + '/pic_zhen'))
        length = int(length / 12)

        for i in range(length):
            image = preprocess(Image.open('data/' + sheet + '/pic_zhen/' + str(i) + '_zoom_' + str(args.zoom) + '_' + args.type + '.png')).unsqueeze(0).to(device)
            ls.append(image)
    ls = torch.concat(ls, dim = 0)

    tensor = model.encode_image(ls)

    return tensor

def get_prompt_feature(args, model, preprocess, device):
    sheet_name = ['恩施市', '利川市', '建始县', '巴东县', '宣恩县', '咸丰县', '来凤县', '鹤峰县']

    prompts = [['mountain', 'plateaus', 'hill', 'plain', 'basin'],
['structural landform', 'erosion landform', 'accumulation landform'],
['crop land', 'garden land', 'forest land', 'grassland'],
['river', 'lake', 'stream', 'pool'],
['town', 'village', 'county', 'city'],
['highway', 'lane', 'avenue', 'pathway'],
['non poverty place', 'poverty place']]

    ls = []
    for sheet in sheet_name:
        length = len(os.listdir('data/' + sheet + '/pic'))
        length = int(length / 12)

        for i in tqdm(range(length)):
            ls_temp = []
            image = preprocess(Image.open('data/' + sheet + '/pic/' + str(i) + '_zoom_' + str(args.zoom) + '_' + args.type + '.png')).unsqueeze(0).to(device)

            for prompt in prompts:
                with torch.no_grad():
                    ls_p = ['a satellite photo of a {}'.format(p) for p in prompt]
                    text = clip.tokenize(ls_p).to(device)

                    logits_per_image, logits_per_text = model(image, text)
                    probs = logits_per_image.softmax(dim=-1).cpu().tolist()
                    ls_temp.extend(probs[0])
            ls.append(ls_temp)

        length = len(os.listdir('data/' + sheet + '/pic_zhen'))
        length = int(length / 12)

        for i in tqdm(range(length)):
            ls_temp = []
            image = preprocess(Image.open('data/' + sheet + '/pic_zhen/' + str(i) + '_zoom_' + str(args.zoom) + '_' + args.type + '.png')).unsqueeze(0).to(device)

            for prompt in prompts:
                with torch.no_grad():
                    ls_p = ['a satellite photo of a {}'.format(p) for p in prompt]
                    text = clip.tokenize(ls_p).to(device)

                    logits_per_image, logits_per_text = model(image, text)
                    probs = logits_per_image.softmax(dim=-1).cpu().tolist()
                    ls_temp.extend(probs[0])
            ls.append(ls_temp)

    return ls
