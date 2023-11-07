import argparse
import math
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam, AdamW, lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
from models.experimental import attempt_load
from models.yolo import Model
from models.yolo import AN_Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.datasets import create_dataloader
from utils.downloads import attempt_download
from utils.general import (LOGGER, check_dataset, check_file, check_git_status, check_img_size, check_requirements,
                           check_suffix, check_yaml, colorstr, get_latest_run, increment_path, init_seeds,
                           intersect_dicts, labels_to_class_weights, labels_to_image_weights, methods, one_cycle,
                           print_args, print_mutation, strip_optimizer)
from utils.loggers import Loggers
from utils.loggers.wandb.wandb_utils import check_wandb_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve, plot_labels
from utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, select_device, torch_distributed_zero_first
from utils.datasets import *
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, skip_nms, print_args, scale_coords, strip_optimizer,
                           xyxy2xywh)
from models.common import *
from tensorboardX import SummaryWriter


def train(hyp, opt, device):
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, \
    freeze = Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
             opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze
    writer = SummaryWriter(log_dir=save_dir)
    w = save_dir / 'weights'  # weights dir
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'  # make dir
    # Hyperparameters
    if isinstance(hyp, str):
        with open(str(ROOT / hyp), errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # Loggers
    data_dict = None
    if RANK in [-1, 0]:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance
        if loggers.wandb:
            data_dict = loggers.wandb.data_dict
            if resume:
                weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size
    cuda = device.type != 'cpu'
    # device = select_device(device)
    init_seeds(1 + RANK)
    weights = str(ROOT / weights)
    pretrained = weights.endswith('.pt') if isinstance(weights, list) else weights.endswith('.pt')
    names = ["car", "truck", "bus", "person", "fire", "smoke", "cone", "div", "suit", "box", "moto", "hat", "ma",
             "bucket"]

    # Trainloader
    nc = 13
    train_path = "/mnt/ecc5c6c9-7631-4983-9ba0-1ec98729589b/Dataset/ancfa/train"
    datafolder_list = ['1k937_070', '2k937_175', '3k1611', '4k925_647', '5k934_518', '6k926_783']
    source_paths = [os.path.join(train_path, n) for n in datafolder_list]
    source = source_paths[-1]
    if pretrained:
        ckpt = torch.load(weights[0], map_location='cpu') if isinstance(weights, list) else torch.load(weights,
                                                                                                       map_location='cpu')
        cfg = "models/Syolov5x.yaml"
        model_an = AN_ST(weights=[weights], device=device, data=source,cfg=str(ROOT/cfg)).to(device)
        # weights_an = ROOT/'1.5last_199.pt'
        # ckpt = torch.load(weights_an,map_location='cpu')
        # model_an = AN_CFA(ckpt=weights_an,device=device,data=source).to(device)

    else:
        model_an = Model(cfg, ch=3, nc=10, anchors=hyp.get('anchors')).to(device)
    stride = model_an.stride
    pt = True
    imgsz = int(opt.imgsz)
    gs = stride
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple
    # dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

    train_loader, dataset = create_dataloader(source, imgsz, batch_size // WORLD_SIZE, gs, single_cls,
                                              hyp=hyp, augment=False, cache=None if opt.cache == 'val' else opt.cache,
                                              rect=opt.rect, rank=LOCAL_RANK, workers=workers,
                                              image_weights=opt.image_weights, quad=opt.quad,
                                              prefix=colorstr('train: '), shuffle=True)
   
    nb = len(train_loader)
    params = [{'params': model_an.parameters()}, ]
    optimizer = AdamW(params=params, lr=5e-4, betas=(0.9, 0.999), weight_decay=1e-3, amsgrad=True)
    # optimizer.load_state_dict(ckpt['optimizer'])

    lf = one_cycle(1, hyp['lrf'], epochs)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    start_epoch = 0
    epochs = 200

    index = 0
    writer = SummaryWriter(log_dir=save_dir)
    for epoch in range(start_epoch, epochs):
        # pbar = enumerate(dataset)
        pbar = enumerate(train_loader)
        pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        # for i, (path, imgs,_, _, _) in pbar:
        for i, (imgs, targets, paths, _) in pbar:
            imgs = imgs.to(device, non_blocking=True).float() / 255
            # imgs = torch.from_numpy(imgs).to(device).float()/ 255.0
            if len(imgs.shape) == 3:
                imgs = imgs[None]  # expand for batch dim

            if len(targets) == 0:
                continue
            optimizer.zero_grad()

            loss= model_an(imgs, targets)
            if loss == None:
                continue
            loss.backward()
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
            pbar.set_description(('%10s' * 2 + '%10.4g' * 1) % (
                f'{epoch}/{epochs - 1}', mem, loss))
            optimizer.step()
            if index % 10 == 0:
                writer.add_scalar('loss', loss.data.cpu(), index)
                writer.add_scalar('optimizer_lr', optimizer.param_groups[0]["lr"], index)
            # if index % 10000 == 0:
            #     ckpt = {
            #         'epoch': epoch,
            #         'model': deepcopy(de_parallel(model_an)),
            #         'optimizer': optimizer.state_dict(),
            #         'date': datetime.now().isoformat()}
            #     past = w/(str(epoch)+'_'+str(i)+'_.pt')
            #     torch.save(ckpt, past)
            index += 1
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()

        ckpt = {'epoch': epoch,
                'model': deepcopy(de_parallel(model_an)),
                'optimizer': optimizer.state_dict(),
                'date': datetime.now().isoformat()}
        past = w / ('last_' + str(epoch) + '.pt')
        torch.save(ckpt, past)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./dlast.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='./models/yolov5x.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='./data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='./data/hyps/hyp_an.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=8, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='AdamW', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=0, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default='runs/st/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[24], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    device = torch.device('cuda:2')
    opt.save_dir = str(increment_path(Path(ROOT / opt.project) / opt.name, mkdir=True))
    train(opt.hyp, opt, device)
