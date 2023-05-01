import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default = 'all', type = str,
                        help = 'Dataset to use')

    parser.add_argument('--zoom', default = 16, type = int,
                        help = 'feature zoom level')

    parser.add_argument('--type', default = 'hybrid', type = str,
                              help = 'image type')

    parser.add_argument('--seed', default = 2022, type = int,
                        help = 'seed for experiment')

    parser.add_argument('--lr', default = 0.05, type = float,
                        help = 'learning rate')

    parser.add_argument('--weight_decay', default = 1e-8, type = float,
                        help = "weight decay for adam optimizer")

    parser.add_argument('--model', default = 'dgcn', type = str,
                        help = 'model selection')

    parser.add_argument('--epoch', default = 1000, type = int,
                        help = 'epoch number')

    parser.add_argument('--patience', default = 10, type = int,
                        help = 'early_stop validation')

    parser.add_argument('--batch_size', default = 1024, type = int,
                        help = 'batch size')

    parser.add_argument('--gpu', default = 0, type = int,
                        help = '-1 for cpu, 0 for gpu:0')

    parser.add_argument('--h_feats', default = 16, type = int,
                        help = 'hidden dimension')

    parser.add_argument('--layers', type = int, default = 1,
                        help = 'convolution layer number')
    parser.add_argument('--village_distance', type = float, default = 5,
                        help = 'max village distance measured in km')
    parser.add_argument('--train_percent', type = float, default = 0.5,
                        help = 'training percent')

    # APPNP param
    parser.add_argument('--k', type = int, default = 10,
                        help = 'Number of propagation steps')
    parser.add_argument('--alpha', type = float, default = 0.9,
                        help = 'Teleport Probability')
    parser.add_argument('--edge-drop', type = float, default = 0.5,
                        help = 'edge propagation dropout')
    parser.add_argument('--in-drop', type = float, default = 0.5,
                        help = 'input feature dropout')

    # gatv2 param
    parser.add_argument('--num_heads', type = int, default = 8,
                        help = 'number of hidden attention heads')
    parser.add_argument('--num_out_heads', type = int, default = 1,
                        help = 'number of output attention heads')
    parser.add_argument('--num_hidden', type = int, default = 8,
                        help = 'number of hidden units')
    parser.add_argument('--attn_drop', type = float, default = 0.0,
                        help = 'attention dropout')
    parser.add_argument('--negative-slope', type = float, default = 0.2,
                        help = 'the negative slope of leaky relu')
    parser.add_argument('--feat_drop', type = float, default = 0.6)

    parser.add_argument('--gamma', type = float, default = 0.2,
                        help = 'trade off between CE and uniformity')
    parser.add_argument('--sigma', type = float, default = 0.2,
                        help = 'trade off between CE and alignment')

    # random walk param
    parser.add_argument('--neighbor_number', type = int, default = 100,
                        help = 'neighborhood number for random walk')
    parser.add_argument('--walkers', type = int, default = 175,
                        help = 'random walker number')
    parser.add_argument('--walk_length', type = int, default = 25,
                        help = 'random walk length')
    parser.add_argument('--h_init', type = int, default = 128,
                        help = 'random walk embedding size')
    parser.add_argument('--window_size', type = int, default = 5,
                        help = 'window size for random walk')
    parser.add_argument('--epochs', type = int, default = 5,
                        help = 'epoch number for centrality2vec training')
    parser.add_argument('--reweight', type = float, default = 0.5,
                        help = 'reweight for category')

    args = parser.parse_args()
    return args
