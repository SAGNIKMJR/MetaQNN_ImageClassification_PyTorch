import argparse

parser = argparse.ArgumentParser(description='PyTorch MetaQNN Image Classification')

# Dataset and loading
parser.add_argument('-train', '--train-data', metavar='TRAINDIR',
                    help='path to (train)dataset')
parser.add_argument('-val', '--val-data', metavar='TESTDIR',
                    help='path to test-dataset')
parser.add_argument('--dataset', default='MNIST',
                    help='name of dataset (options: MNIST, CIFAR10, CIFAR100)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-p', '--patch-size', default=28, type=int,
                    metavar='P', help='patch size for crops')

# Continue from qstore
parser.add_argument('-qstp', '--qstore-path', metavar='QSTP', default = None,
                    help='path to stored q values (default: None)')
parser.add_argument('-rdp', '--replay-dict-path', metavar='RDP', default = None,
                    help='path to replay dictionary (default: None)')
parser.add_argument('-ce', '--continue-epsilon', default=1.0, type=float,
                    metavar='CE', help='epsilon value to continue from (default: 1.0)')
parser.add_argument('-ci', '--continue-ite', default=1, type=int,
                    metavar='CI', help='iteration no. to continue from, should be' 
                    'lesser than limit in epsilon schedule (default: 1)')


# Model
parser.add_argument('--weight-init', default='kaiming-normal', metavar='W',
                    help='weight-initialization scheme (default: kaiming-normal)')

# Training hyper-parameters
parser.add_argument('--no-gpus', default=1, type=int, metavar='G',
                    help='number of gpus to run the code on')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default 0.9)')
parser.add_argument('-wd', '--weight-decay', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)')
parser.add_argument('-bn', '--batch-norm', default=1e-5, type=float,
                    metavar='BN', help='batch normalization (default 1e-5)')
parser.add_argument('-dod', '--drop-out-drop', default=0.2, type=float,
                    metavar='DOD', help='drop out drop probability (default 0.2)')
parser.add_argument('-pf', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')

# Learning Rate schedule
parser.add_argument('-iu', '--init-utility', default=0.50, type=float,
                    metavar='IU', help='initial utility of all transitions, set to acc of avg model\
                     (default: 0.50)')
parser.add_argument('-lr', '--learning-rate', default=1e-1, type=float,
                    metavar='LR', help='initial learning rate (default: 0.1)')
parser.add_argument('--lr-wr-epochs', default=10, type=int,
                    help='epochs defining one warm restart cycle')
parser.add_argument('--lr-wr-mul', default=2, type=int,
                    help='factor to increase warm restart cycle epochs after restart')
parser.add_argument('--lr-wr-min', default=1e-5, type=float,
                    help='minimum learning rate used in each warm restart cycle')

# QLearning parameters
parser.add_argument('-qlr', '--q-learning-rate', default=0.1, type=float,
                    help='Q update learning rate if actual q-learning update rule used')
parser.add_argument('-qdis', '--q-discount-factor', default=1.0, type=float,
                    help='Q learning discount factor')
parser.add_argument('--conv-layer-max-limit', default=6, type=int,
                    help='Maximum amount of conv layers in model (default: 6)')
parser.add_argument('--conv-layer-min-limit', default=1, type=int,
                    help='Minimum amount of conv layers in model (default: 3)')
parser.add_argument('--max-fc', default=1, type=int,
                    help='maximum amount of hidden fully-connected layers (default: 1),\
                         sigmoid layer is not being counted as an FC layer here')
