import dgl
import dgl.function as fn
from tqdm import tqdm
import pdb
import numpy as np
import torch
import torch.nn as nn
import logging
from utils.parser import parse_args
from utils.utils import config, choose_dataset, choose_model, get_clip_feature, centrality2vec
import torch.nn.functional as F
import logging
from torchmetrics.functional import accuracy, precision, recall, f1_score
from torchmetrics import AUROC
import clip
from dgl.nn import LabelPropagation

if __name__ == '__main__':
    args = parse_args()
    early_stop = config(args)

    if args.gpu >= 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'
    device = torch.device(device)
    args.device = device

    dataset = choose_dataset(args)
    g = dataset[0].to(device)

    feat = torch.tensor(centrality2vec(args)).to(device)


    model = choose_model(args, feat.shape[1], dataset.num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)

    early_stop(0.0, model)

    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    reweight = args.reweight
    weight = torch.tensor([reweight, 1 - reweight], dtype = torch.float).to(device)

    for e in range(args.epoch):

        logits = model(g, feat)

        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss = F.cross_entropy(logits[train_mask], labels[train_mask], weight = weight)


        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        logging.info('train acc = {}'.format(train_acc.item()))
        early_stop(val_acc, model)

        if early_stop.early_stop:
            break

        loss.backward()
        opt.step()
        opt.zero_grad()

    logging.info('loading best model for test')
    model.load_state_dict(torch.load(early_stop.save_path))
    model.eval()
    logits = model(g, feat)
    pred = logits.argmax(1)[test_mask]

    gt = labels[test_mask]

    auroc = AUROC(pos_label = 1)

    logging.info('test acc_1 = {}'.format(accuracy(pred, gt).item()))
    logging.info('test precision_1 = {}'.format(precision(pred, gt, average = None, num_classes = dataset.num_classes)[1]))
    logging.info('test recall_1 = {}'.format(recall(pred, gt, average = None, num_classes = dataset.num_classes)[1]))
    logging.info('test f1_1 = {}'.format(f1_score(pred, gt, average = None, num_classes = dataset.num_classes)[1]))
    logging.info('test auroc = {}'.format(auroc(F.softmax(logits, dim = 1)[:, 1][test_mask], gt)))
    logging.info('===================================================')

    logging.info('test acc = {}'.format(accuracy(pred, gt).item()))
    logging.info('test precision = {}'.format(precision(pred, gt, average = 'macro', num_classes = dataset.num_classes)))
    logging.info('test recall = {}'.format(recall(pred, gt, average = 'macro', num_classes = dataset.num_classes)))
    logging.info('test f1 = {}'.format(f1_score(pred, gt, average = 'macro', num_classes = dataset.num_classes)))
    logging.info('test auroc = {}'.format(auroc(F.softmax(logits, dim = 1)[:, 1][test_mask], gt)))



