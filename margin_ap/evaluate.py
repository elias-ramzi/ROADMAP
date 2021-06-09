import logging
import argparse

import torch

from margin_ap.getter import Getter
import margin_ap.utils as lib
import margin_ap.engine as eng


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--set", type=str, default='test')
parser.add_argument("--bs", type=str, default=128)
parser.add_argument("--nw", type=str, default=10)
parser.add_argument("--data_dir", type=str, default=None)
args = parser.parse_args()

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO,
)

state = torch.load(lib.expand_path(args.config), map_location='cpu')
cfg = state["config"]

logging.info("Loading model...")
net = Getter().get_model(cfg.model)
net.load_state_dict(state["net_state"])
if torch.cuda.device_count() > 1:
    net = torch.nn.DataParallel(net)
net.cuda()

# cfg.dataset.kwargs.data_dir = fix_path(cfg.dataset.kwargs.data_dir)

transform = Getter().get_transform(cfg.transform.test)
try:
    dts = Getter().get_dataset(transform, args.set, cfg.dataset)
except FileNotFoundError as exc:
    if args.data_dir is not None:
        cfg.dataset.kwargs.data_dir = lib.expand_path(args.data_dir)
        dts = Getter().get_dataset(transform, args.set, cfg.dataset)
    else:
        raise exc
logging.info("Dataset created...")

metrics = eng.evaluate(
    net=net,
    test_dataset=dts,
    epoch='0',
    batch_size=args.bs,
    num_workers=args.nw,
)


logging.info("Evaluation completed...")
for split, mtrc in metrics.items():
    for k, v in mtrc.items():
        if k == 'epoch':
            continue
        logging.info(f"{split} --> {k} : {v:.4f}")
