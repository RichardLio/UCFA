import argparse
import math
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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
from sklearn.metrics import roc_auc_score, auc
import numpy as np
from skimage.measure import label, regionprops
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from scipy.ndimage import gaussian_filter
from skimage import morphology
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
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
                           increment_path, non_max_suppression, skip_nms,print_args, scale_coords, strip_optimizer, xyxy2xywh)
from models.common import*
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
        with open(str(ROOT/hyp), errors='ignore') as f:
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
    c_w = 'C_nup.pth'
    des_w = 'descriptor.pth'
    weights = [str(ROOT/weights),str(ROOT/c_w),str(ROOT/des_w)]
    weights  =  weights [0]
    pretrained = weights[0].endswith('.pt') if isinstance(weights,list) else  weights.endswith('.pt')
    names = ["car","truck","bus","person","fire","smoke","cone","div","suit","box","moto","hat","ma","bucket"]
    
     # Trainloader
    nc = 13
    train_path ="/mnt/ecc5c6c9-7631-4983-9ba0-1ec98729589b/Dataset/ancfa/train"
    datafolder_list = ['1k937_070', '2k937_175','3k1611','4k925_647','5k934_518','6k926_783']
    source_paths =[os.path.join(train_path,n) for n in datafolder_list] 
    source = source_paths[0]
    if pretrained:
        ckpt = torch.load(weights[0], map_location='cpu') if isinstance(weights,list) else torch.load(weights, map_location='cpu') 
        model_an = AN_CFA_V2(weights=weights,device = device,data=source).to(device)
        # weights_an = ROOT/'1.5last_199.pt'
        # ckpt = torch.load(weights_an,map_location='cpu')
        # model_an = AN_CFA(ckpt=weights_an,device=device,data=source).to(device)
        
    else:
        model_an = Model(cfg, ch=3, nc=10, anchors=hyp.get('anchors')).to(device) 
    stride =  model_an.stride
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
    model_an._init_centroid(train_loader,save_dir= save_dir)
    # center_loader, _ = create_dataloader(train_path, imgsz, 1, gs, single_cls,
    #                                           hyp=hyp, augment=False, cache=None if opt.cache == 'val' else opt.cache,
    #                                           rect=opt.rect, rank=LOCAL_RANK, workers=workers,
    #                                           image_weights=opt.image_weights, quad=opt.quad,
    #                                           prefix=colorstr('train: '), shuffle=True)
    # model_an._init_centroid(center_loader)
    # mlc = int(np.concatenate(dataset.labels, 0)[:, 0].max())  # max label class
    # nb = len(train_loader)  # number of batches
    # assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'
    nb = len(train_loader)
    params = [{'params' : model_an.parameters()},]
    optimizer = AdamW(params = params, lr = 1e-3,betas=(0.9, 0.999), amsgrad= True) 
    # optimizer.load_state_dict(ckpt['optimizer'])
   
    lf = one_cycle(1, hyp['lrf'], epochs) 
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    start_epoch = 0
    epochs = 100
    
    index = 0 
    writer = SummaryWriter(log_dir=save_dir)
    source_an = "/mnt/ecc5c6c9-7631-4983-9ba0-1ec98729589b/Dataset/ancfa/test/1k937_070"
    source_mask = "/mnt/ecc5c6c9-7631-4983-9ba0-1ec98729589b/Dataset/ancfa/test/1k937_070/mask"
    best_img_roc = 1e-5
    best_pxl_roc = 1e-5
    best_pxl_pro = 1e-5
    best_threshold = 1e-5
    best_values = 1e-5
    img_roc_auc = 1e-5
    per_pixel_rocauc = 1e-5
    per_pixel_proauc = 1e-5
    threshold = 1e-5
    for epoch in range(start_epoch, epochs): 
        #pbar = enumerate(dataset)
        pbar = enumerate(train_loader)
        pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') 
        # for i, (path, imgs,_, _, _) in pbar: 
        for i, (imgs, targets, paths, _) in pbar: 
            imgs = imgs.to(device, non_blocking=True).float() / 255
            # imgs = torch.from_numpy(imgs).to(device).float()/ 255.0
            optimizer.zero_grad() 
            if len(imgs.shape) == 3:
                imgs = imgs[None]  # expand for batch dim
            if len(targets) == 0:
                continue
            loss , L_att , L_rep = model_an(imgs,targets,[ epoch,epochs])
            if loss ==None:
                continue
            loss.backward()
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
            pbar.set_description(('%10s' * 2 + '%10.4g' * 1) % (
                    f'{epoch}/{epochs - 1}', mem, loss))
            optimizer.step()
            if index % (200) == 0:
                writer.add_scalar('loss', loss.data.cpu(), index)
                writer.add_scalar('l_att', L_att.data.cpu(), index)
                writer.add_scalar('l_rep', L_rep.data.cpu(), index)
                writer.add_scalar('r', model_an.r.cpu(), index)
                writer.add_scalar('optimizer_lr', optimizer.param_groups[0]["lr"], index)
                writer.add_scalar('best_img_roc', best_img_roc, index)
                writer.add_scalar('best_pxl_roc', best_pxl_roc, index)
                writer.add_scalar('best_pxl_pro', best_pxl_pro, index)
                writer.add_scalar('img_roc_auc', img_roc_auc, index)
                writer.add_scalar('per_pixel_rocauc', per_pixel_rocauc, index)
                writer.add_scalar('per_pixel_proauc', per_pixel_proauc, index)
                writer.add_scalar('best_threshold', best_threshold, index)
                writer.add_scalar('threshold', threshold, index)
                writer.add_scalar('best_values', best_values,  index)
                
            index += 1
        # writer.add_scalar('loss', loss.data.cpu(),  epoch)
        # writer.add_scalar('l_att', L_att.data.cpu(),  epoch)
        # writer.add_scalar('l_rep', L_rep.data.cpu(),  epoch)
        # writer.add_scalar('r', model_an.r.cpu(),  epoch)
        # writer.add_scalar('optimizer_lr', optimizer.param_groups[0]["lr"],  epoch)
        # writer.add_scalar('best_img_roc', best_img_roc,  epoch)
        # writer.add_scalar('best_pxl_roc', best_pxl_roc,  epoch)
        # writer.add_scalar('best_pxl_pro', best_pxl_pro,  epoch)
        # writer.add_scalar('img_roc_auc', img_roc_auc,  epoch)
        # writer.add_scalar('per_pixel_rocauc', per_pixel_rocauc,  epoch)
        # writer.add_scalar('per_pixel_proauc', per_pixel_proauc,  epoch)
        # writer.add_scalar('best_threshold', best_threshold,  epoch)
        # writer.add_scalar('threshold', threshold,  epoch)
        # writer.add_scalar('best_values', best_values,  epoch)

        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        # scheduler.step()
        
        ckpt = {'epoch': epoch,
                'model': deepcopy(de_parallel(model_an)),
                'optimizer': optimizer.state_dict(),
                'date': datetime.now().isoformat()}
        past = w/('last_'+str(epoch)+'.pt')
        torch.save(ckpt, past)
        best_list = [best_img_roc,best_pxl_roc,best_pxl_pro,best_threshold, best_values]
        best_img_roc, best_pxl_roc,\
        best_pxl_pro,best_threshold,\
            img_roc_auc,per_pixel_rocauc,\
            per_pixel_proauc,threshold,best_values  =eval(model_an,device,source_an,source_mask ,imgsz,stride,save_dir,epoch,best_list)
        # path_all_collects = []
        # path_ori_collects = []
        # for j in range(20):
        #     path, im, _, _, _ = dataset.__next__()
        #     heatmap_an,img_ins_an,save_names_an, p_an = get_score(im,model_an,path,device)
        #     if heatmap_an is None:
        #         continue
        #     path_all = []
        #     path_ori = []
        #     for n in save_names_an:
        #         path_all.append(str(j)+'_an_' +n + '.jpg' )
        #         path_ori.append(str(j)+'_o_' +n + '.jpg')
        #         path_all_collects.append(str(j)+'_anall_' +n + '.jpg' )
        #         path_ori_collects.append(str(j)+'_oriall_' +n + '.jpg' )
        #     if len(heatmaps_scores) == 0  :
        #         heatmaps_scores = heatmap_an
        #     else:
        #         heatmaps_scores = np.concatenate((heatmaps_scores, heatmap_an), axis=0)
        #     img_collects = torch.cat((img_collects, img_ins_an ), dim=0) if img_collects != None else img_ins_an
        #     for k in range(len(img_ins_an )):
        #         hp = heatmap_an[k]
        #         imgi = img_ins_an[k]
        #         hp = (hp - heatmap_an.min()) / (heatmap_an.max() - heatmap_an.min())
        #         hp *= 255.0
        #         imgi = np.uint8(imgi.permute(1,2,0).numpy()[:,:,[2,1,0]])
        #         hp = cv2.applyColorMap(np.uint8(hp), cv2.COLORMAP_JET)
        #         visual_map = (hp / 255.0 + imgi / 255.0) * 255.0
        #         visual_map = (visual_map / visual_map.max()) * 255.0
                
        #         newsave_dir = str(save_dir)+'/detect_'+str(epoch)+'_'+str(round(float(model_an.r.data),3))
        #         newsave_dir = Path(newsave_dir )
        #         if not newsave_dir.exists():
        #             newsave_dir.mkdir(parents=True, exist_ok=True) 
        #         save_an_path = str(newsave_dir/path_all[k])
        #         save_img_pah = str(newsave_dir/path_ori[k])
        #         print('save ',save_an_path)
        #         print('save ', save_img_pah)
        #         cv2.imwrite(save_an_path , np.uint8(visual_map))
        #         cv2.imwrite(save_img_pah, np.uint8(imgi))
        
       
def get_score(im,model,path,device):
    im = torch.from_numpy(im).to(device)
    im = im.float() 
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    # detectbox = torch.tensor([[0.83333, 0.34531, 0.91146, 0.41302]])# 3k
    # detectbox = torch.tensor([[0.30900,0.37937,0.37179,0.42593]]) #4k
    # detectbox = torch.tensor([[0.71563, 0.27969, 0.74531, 0.30625]]) #5k
    # detectbox = torch.tensor([[0.50781, 0.40417, 0.83646, 0.78073]]) #6k
    detectbox = torch.tensor([[0.43854, 0.40156, 0.61458, 0.66979]]) # 1k
    heatmap,img_ins,save_names = model(im,detectbox,[])
    if heatmap ==None:
        return None,None,None,None
    # heatmap = heatmap.numpy()
    heatmap *= 255.0
    p = Path(path)
    p = p.name
    return heatmap,img_ins,save_names, p

def eval(model,device,source_an,source_mask ,imgsz,stride,save_dir,epoch,best):
    model.detect = True
    dataset_an = LoadImages(source_an, img_size=imgsz, stride=stride, auto=True)
    dataset_mask = LoadImages(source_mask, img_size=imgsz, stride=stride, auto=True)
    dataset_an.__iter__()
    dataset_mask.__iter__()
    path_all_collects = []
    path_ori_collects = []
    img_collects = None
    img_masks_collects = None
    heatmaps_scores  =[]
    best_sum =0
    for b in best:
        best_sum+= b
    if best_sum>4e-4:
        best_img_roc,best_pxl_roc,best_pxl_pro,best_threshold,best_values  = best
    else:
        best_img_roc = 1e-5
        best_pxl_roc = 1e-5
        best_pxl_pro = 1e-5
        best_threshold = 1e-5
        best_values = 1e-5
    
    for i in range(len(dataset_mask)):
        path_an, im_an, _, _, _ =dataset_an.__next__()
        path_mask, im_mask, _, _, _ =dataset_mask.__next__()
       
        heatmap_an,img_ins_an,save_names_an, p_an = get_score(im_an,model,path_an,device)
        img_mask = get_mask(im_mask,device)
        if heatmap_an == None :
            continue
        img_ins_all = img_ins_an
        heatmap_all = heatmap_an
        heatmap_all = heatmap_all.numpy()
        for n in save_names_an:
            path_all_collects .append(str(i)+'_an_' +n + '.jpg' )
            path_ori_collects.append(str(i)+'_o_' +n + '.jpg')
        if len(heatmaps_scores) == 0  :
            heatmaps_scores = heatmap_all
        else:
            heatmaps_scores = np.concatenate((heatmaps_scores, heatmap_all), axis=0)
       
        img_collects = torch.cat((img_collects, img_ins_all ), dim=0) if img_collects != None else img_ins_all 
        img_masks_collects = torch.cat((img_masks_collects, img_mask ), dim=0) if img_masks_collects != None else img_mask
        if len(img_collects) >=len(dataset_mask) :
           
            masks_data =  img_masks_collects.cpu().numpy()
            masks = np.stack([cv2.cvtColor(m.transpose(1,2,0),cv2.COLOR_RGB2GRAY) for m in masks_data],axis=0)
            masks = np.int32((masks>=255))
            hp_scores = (heatmaps_scores - heatmaps_scores.min()) / (heatmaps_scores.max() - heatmaps_scores.min())
            
            threshold = get_threshold(masks.copy(), hp_scores)
            
            gt_list = []
            gt_list.extend([1] * len(masks))
            gt_list[0] = 0
            fpr, tpr, img_roc_auc = cal_img_roc( hp_scores.copy(),gt_list)
            best_img_roc = img_roc_auc if img_roc_auc > best_img_roc else best_img_roc
            fpr, tpr, per_pixel_rocauc = cal_pxl_roc(masks.copy(), hp_scores.copy())
            best_pxl_roc = per_pixel_rocauc if per_pixel_rocauc > best_pxl_roc else best_pxl_roc
            
            # r'Pixel-level AUPRO'
            per_pixel_proauc = cal_pxl_pro(masks.copy(), hp_scores.copy())
            best_pxl_pro = per_pixel_proauc if per_pixel_proauc > best_pxl_pro else best_pxl_pro
            
            print('image ROCAUC: %.3f | best: %.3f'% ( img_roc_auc , best_img_roc))
            print('pixel ROCAUC: %.3f | best: %.3f'% ( per_pixel_rocauc, best_pxl_roc))
            print('pixel PROAUC: %.3f | best: %.3f'% ( per_pixel_proauc, best_pxl_pro))
            newsave_dir = str(save_dir)+'/detect_'+str(epoch)+'_'+str(round(float(model.r.data),3))
            newsave_dir = Path(newsave_dir )
            if not newsave_dir.exists():
                    newsave_dir.mkdir(parents=True, exist_ok=True)
            txt_save_name = str(newsave_dir)+'/'
            img_roc_txt = txt_save_name + 'img_roc:'+str(round(float(img_roc_auc),3))+ '_'+'best:'+str(round(float(best_img_roc),3))+'.txt'
            pxl_roc_txt = txt_save_name + 'pxl_roc:'+str(round(float(per_pixel_rocauc),3))+ '_'+'best:'+str(round(float(best_pxl_roc),3))+'.txt'
            pxl_pro_txt = txt_save_name + 'pxl_pro:'+str(round(float(per_pixel_proauc),3))+ '_'+'best:'+str(round(float(best_pxl_pro),3))+'.txt'
            thre_txt = txt_save_name + 'thre'+str(round(float(threshold),3))+ '_'+'best:'+str(round(float(best_threshold),3))+'.txt'
            mean_values = (img_roc_auc + per_pixel_rocauc + per_pixel_proauc)/3
            best_values = mean_values if mean_values> best_values else best_values
            if mean_values>= best_values:
                best_txt = str(save_dir)+'/best_'+'epoch:'+str(epoch)+'_' + \
                    'img_roc:'+str(round(float(img_roc_auc),3))+ '_'+\
                        'pxl_roc:'+str(round(float(per_pixel_rocauc),3))+\
                         'pxl_pro:'+str(round(float(per_pixel_proauc),3))+\
                          'thre:'+ str(round(float(threshold),3))  +'.txt'
                np.savetxt(best_txt,np.zeros(0))
                print('mean best values--img_roc: %.3f |pxl_roc: %.3f|pxl_pro:%.3f|threshold:%.3f '%(img_roc_auc,per_pixel_rocauc,per_pixel_proauc,threshold))
            np.savetxt(img_roc_txt,np.zeros(0))
            np.savetxt(pxl_roc_txt,np.zeros(0))
            np.savetxt(pxl_pro_txt,np.zeros(0))
            np.savetxt(thre_txt,np.zeros(0))
            plot_fig(img_collects.clone(), hp_scores.copy(), masks_data.copy(), threshold, newsave_dir)
            # for j in range(len(img_collects )):
            #     hp = hp_scores[j]
            #     imgi = img_collects[j]
            #     hp *= 255.0
            #     imgi = np.uint8(imgi.permute(1,2,0).numpy()[:,:,[2,1,0]])
            #     hp = cv2.applyColorMap(np.uint8(hp), cv2.COLORMAP_JET)
            #     visual_map = (hp / 255.0 + imgi / 255.0) * 255.0
            #     visual_map = (visual_map / visual_map.max()) * 255.0
            #     # newsave_dir = str(save_dir)+'/detect_'+str(epoch)+'_'+str(round(float(model.r.data),3))
            #     # newsave_dir = Path(newsave_dir )
            #     # if not newsave_dir.exists():
            #     #     newsave_dir.mkdir(parents=True, exist_ok=True) 
            #     save_an_path = str(newsave_dir/path_all_collects[j])
            #     save_img_pah = str(newsave_dir/path_ori_collects[j])
            
            #     print('save ',save_an_path)
            #     print('save ', save_img_pah)
            #     cv2.imwrite(save_an_path , np.uint8(visual_map))
            #     cv2.imwrite(save_img_pah, np.uint8(imgi))
    model.detect = False
    return best_img_roc, best_pxl_roc,best_pxl_pro,best_threshold,\
        img_roc_auc,per_pixel_rocauc,per_pixel_proauc,threshold,best_values 

def get_mask(im,device):
    
    im = torch.from_numpy(im).to(device)
    im = im.float() 
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    #detectbox = torch.tensor([[0.83333, 0.34531, 0.91146, 0.41302]]) # 3k
    # detectbox = torch.tensor([[0.30900,0.37937,0.37179,0.42593]]) # 4k
    # detectbox = torch.tensor([[0.71563, 0.27969, 0.74531, 0.30625]])# 5k
    # detectbox = torch.tensor([[0.50781, 0.40417, 0.83646, 0.78073]]) # 6k
    detectbox = torch.tensor([[0.43854, 0.40156, 0.61458, 0.66979]]) # 1k
    detectbox *= torch.tensor([640,640,640,640])
    
    detectbox = detectbox.to(device)
    boxes  = detectbox[:,:4]
    wh =  boxes[:,2:] - boxes[:,:2]
    cxcy = boxes[:,:2] + wh/2
    newboxes = torch.cat([cxcy, 1.20*wh],dim=-1)
    newboxes[:,:2] -= newboxes[:,2:]/2
    newboxes[:, 2:] += newboxes[:,:2]
    newboxes = newboxes.clamp(max=640-1,min=0.0)
    newboxes[:,3] = newboxes[:,3].clamp(max=500-1,min=0.0)
    boxes = newboxes
    img_ins,save_names = roi_img(im/255.0,[boxes],(128,128),(640,640))
    return img_ins.cpu()

def roc_auc_img(gt, score):
    img_roc_auc = roc_auc_score(gt, score)
    
    return img_roc_auc

def roc_auc_pxl(gt, score):
    per_pixel_roc_auc = roc_auc_score(gt.flatten(), score.flatten())

    return per_pixel_roc_auc


def pro_auc_pxl(gt, score):
    # gt = np.squeeze(gt, axis=1)
    gt[gt <= 0.5] = 0
    gt[gt > 0.5] = 1
    gt = gt.astype(np.bool_)

    max_step = 200
    expect_fpr = 0.3

    max_th = score.max()
    min_th = score.min()
    delta = (max_th - min_th) / max_step

    pros_mean = []
    pros_std = []
    threds = []
    fprs = []

    binary_score_maps = np.zeros_like(score, dtype=np.bool_)

    for step in range(max_step):
        thred = max_th - step * delta
        binary_score_maps[score <= thred] = 0
        binary_score_maps[score > thred] = 1

        pro = []
        for i in range(len(binary_score_maps)):
            label_map = label(gt[i], connectivity=2)
            props = regionprops(label_map, binary_score_maps[i])
            
            for prop in props:
                pro.append(prop.intensity_image.sum() / prop.area)

        pros_mean.append(np.array(pro).mean())
        pros_std.append(np.array(pro).std())

        gt_neg = ~gt
        fpr = np.logical_and(gt_neg, binary_score_maps).sum() / gt_neg.sum()
        fprs.append(fpr)
        threds.append(thred)

    threds = np.array(threds)
    pros_mean = np.array(pros_mean)
    pros_std = np.array(pros_std)
    fprs = np.array(fprs)

    idx = fprs <= expect_fpr 
    fprs_selected = fprs[idx]
    fprs_selected = rescale(fprs_selected)
    pros_mean_selected = rescale(pros_mean[idx])
    per_pixel_roc_auc = auc(fprs_selected, pros_mean_selected)

    return per_pixel_roc_auc


def rescale(x):
        return (x - x.min()) / (x.max() - x.min())
def get_threshold(gt, score):
    gt_mask = np.asarray(gt)
    precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), score.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]

    return threshold

def gaussian_smooth(x, sigma=4):
    bs = x.shape[0]
    for i in range(0, bs):
        x[i] = gaussian_filter(x[i], sigma=sigma)

    return x

def upsample(x, size, mode):
    return F.interpolate(x.unsqueeze(1), size=size, mode=mode, align_corners=False).squeeze().numpy()

def cal_img_roc(scores, gt_list):
    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
    gt_list = np.asarray(gt_list)
    fpr, tpr, _ = roc_curve(gt_list, img_scores)
    img_roc_auc = roc_auc_img(gt_list, img_scores)

    return fpr, tpr, img_roc_auc

def cal_pxl_roc(gt_mask, scores):
    fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
    per_pixel_rocauc = roc_auc_pxl(gt_mask.flatten(), scores.flatten())
    
    return fpr, tpr, per_pixel_rocauc

def cal_pxl_pro(gt_mask, scores):
    per_pixel_proauc = pro_auc_pxl(gt_mask, scores)

    return per_pixel_proauc

def plot_fig(test_img, scores, gts, threshold, save_dir):
    num = len(scores)
    for i in range(num):
        img = test_img[i]
        img = np.uint8(img.permute(1,2,0).numpy()[:,:,[2,1,0]])
        gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(2.75
                                 )
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        
        fig = plt.figure()
        ax0 = fig.add_subplot(221)
        ax0.axis('off')
        ax0.imshow(img)
        ax0.title.set_text('Image')

        ax1 = fig.add_subplot(222)
        ax1.axis('off')
        ax1.imshow(gt.mean(axis=-1), cmap='gray')
        # ax1.imshow(gt, cmap='gray')
        ax1.title.set_text('GroundTruth')
        
        ax2 = fig.add_subplot(223)
        ax2.axis('off')
        ax2.imshow(img, cmap='gray', interpolation='none')
        ax2.imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax2.title.set_text('Predicted heat map')

        ax3 = fig.add_subplot(224)
        ax3.axis('off')
        ax3.imshow(vis_img)
        ax3.title.set_text('Segmentation result')

        fig.tight_layout()
        fig.savefig(os.path.join(save_dir,'an_cars' + '_{}'.format(i)), dpi=100)
        plt.close()


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./dlast.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='./models/yolov5x.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='./data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='./data/hyps/hyp_an.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs, -1 for autobatch')
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
    parser.add_argument('--project', default='runs/cfabn/train', help='save to project/name')
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
    opt.save_dir = str(increment_path(Path(ROOT/opt.project) / opt.name,mkdir=True))
    train(opt.hyp, opt, device)
