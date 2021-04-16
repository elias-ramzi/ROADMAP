from argparse import ArgumentParser

from .do_train import do_train


parser = ArgumentParser()
# """""""""""""""""""" Experience """"""""""""""""""""
parser.add_argument('--log_dir', type=str, required=True)
parser.add_argument('--experiment_name', type=str, required=True)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--max_iter', type=int, default=50)
parser.add_argument('--resume', type=str, default=None)

# """""""""""""""""""" Data """"""""""""""""""""
parser.add_argument('--data_dir', type=str, default="/local/DEEPLEARNING/image_retrieval")
parser.add_argument('--dataset', type=str, default="sop")
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--batches_per_super_pair', type=int, default=10)
parser.add_argument('--val_bs', type=int, default=96)
parser.add_argument('--val_freq', type=int, default=5)
parser.add_argument('--num_workers', type=int, default=10)

# """""""""""""""""""" Model """"""""""""""""""""
parser.add_argument('--backbone', type=str, default="resnet50")
parser.add_argument('--embed_dim', type=int, default=512)
parser.add_argument('--norm_features', action='store_true', default=False)
parser.add_argument('--without_fc', action='store_true', default=False)
parser.add_argument('--freeze_bn', action='store_false', default=True)

# """""""""""""""""""" Optimizer """"""""""""""""""""
parser.add_argument('--optimizer', type=str, default="adam")
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--wd', type=float, default=1e-4)
parser.add_argument('--nlr', type=float, default=1e-4)
parser.add_argument('--nwd', type=float, default=1e-4)

# """""""""""""""""""" Scheduler """"""""""""""""""""
parser.add_argument('--scheduler', type=str, default="step")
parser.add_argument('--milestones', type=int, nargs='+', default=None)
parser.add_argument('--gamma', type=float, default=None)

# """""""""""""""""""" Loss """"""""""""""""""""
parser.add_argument('--criterion', type=str, default="smoothap")
parser.add_argument('--temp', type=float, default=0.01)
parser.add_argument('--mu', type=float, default=0.05)
parser.add_argument('--tau', type=float, default=0.05)


args = parser.parse_args()

metrics = do_train(**args)
