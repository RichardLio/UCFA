import argparse
import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from scipy.ndimage import gaussian_filter
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
# from models.common import DetectMultiBackend
from utils.datasets import *
# from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams, LoadMyImages
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, skip_nms,print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from models.common import *
from sklearn.metrics import roc_auc_score, auc
import numpy as np
from skimage.measure import label, regionprops
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from scipy.ndimage import gaussian_filter
import torch.nn.functional as F
from skimage import morphology
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5x.pt', help='model path(s)')
    parser.add_argument('--weights_an', nargs='+', type=str, default=ROOT / '4knobn1.2_3l_199.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/cfa_detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt

@torch.no_grad()
def run(weights=ROOT / 'yolov5x.pt',  # model.pt path(s)
        weights_an = ROOT/'bn4k_1.2.pt',
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.1,  # confidence threshold
        iou_thres=0.3,  # NMS IOU threshold
        max_det=100,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/cfa_detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        ):
    source = str(source)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = torch.device('cpu')
    model = AN_CFA_V2(ckpt=weights_an,detect=True,device=device)
    model.descriptor.eval()
    stride, names = model.stride, model.names
    names = ["car","truck","bus","person","fire","smoke","cone","div","suit","box","moto","hat","ma","bucket"]
    pt = True
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    source_an = "/mnt/ecc5c6c9-7631-4983-9ba0-1ec98729589b/Dataset/ancfa/test/4k925_647"
    source = "/mnt/ecc5c6c9-7631-4983-9ba0-1ec98729589b/Dataset/ancfa/train/4k925_647"
    source_mask = "/mnt/ecc5c6c9-7631-4983-9ba0-1ec98729589b/Dataset/ancfa/test/4k925_647/mask"
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    dataset_an = LoadImages(source_an, img_size=imgsz, stride=stride, auto=pt)
    dataset_mask = LoadImages(source_mask, img_size=imgsz, stride=stride, auto=pt)
    dataset.__iter__()
    dataset_an.__iter__()
    dataset_mask.__iter__()
    bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    heatmaps_scores = []
    count = 0
    img_collects = None
    img_masks_collects = None
    path_all = []
    path_ori = []
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]
    for i in range(len(dataset_mask)):
        # path, im, _, _, _ = dataset.__next__()
        path_an, im_an, _, _, _ =dataset_an.__next__()
        path_mask, im_mask, _, _, _ =dataset_mask.__next__()
       
        heatmap_an,img_ins_an,save_names_an, p_an = get_score(im_an,model,path_an,device)
        img_mask = get_mask(im_mask,device)
        if heatmap_an == None :
            continue
        # if heatmap_an == None or heatmap ==None:
        #     continue
        
        img_ins_all = img_ins_an
        heatmap_all = heatmap_an
        # path_all = []
        # path_ori = []
        heatmap_all = heatmap_all.numpy()
        for n in save_names_an:
            path_all.append(str(i)+'_an_' +n + '.jpg' )
            path_ori.append(str(i)+'_o_' +n + '.jpg')
        if len(heatmaps_scores) == 0  :
            heatmaps_scores = heatmap_all
        else:
            heatmaps_scores = np.concatenate((heatmaps_scores, heatmap_all), axis=0)
       
        img_collects = torch.cat((img_collects, img_ins_all ), dim=0) if img_collects != None else img_ins_all 
        img_masks_collects = torch.cat((img_masks_collects, img_mask ), dim=0) if img_masks_collects != None else img_mask
        if len(img_collects) >=10 :
           
            masks =  img_masks_collects.cpu().numpy()
            masks = np.stack([cv2.cvtColor(m.transpose(1,2,0),cv2.COLOR_RGB2GRAY) for m in masks],axis=0)
            masks = np.int32((masks>=255))
            hp_scores = (heatmaps_scores - heatmaps_scores.min()) / (heatmaps_scores.max() - heatmaps_scores.min())
            
            threshold = get_threshold(masks.copy(), hp_scores)
            gt_list = []
            gt_list.extend([0] * len(masks))
            gt_list[0] = 1
            fpr, tpr, img_roc_auc = cal_img_roc( hp_scores.copy(),gt_list)
            best_img_roc = img_roc_auc 
            fpr, tpr, per_pixel_rocauc = cal_pxl_roc(masks.copy(), hp_scores.copy())
            best_pxl_roc = per_pixel_rocauc 
            
            # r'Pixel-level AUPRO'
            per_pixel_proauc = cal_pxl_pro(masks, hp_scores.copy())
            best_pxl_pro = per_pixel_proauc
            
            print('image ROCAUC: %.3f | best: %.3f'% ( img_roc_auc , best_img_roc))
            print('pixel ROCAUC: %.3f | best: %.3f'% ( per_pixel_rocauc, best_pxl_roc))
            print('pixel PROAUC: %.3f | best: %.3f'% ( per_pixel_proauc, best_pxl_pro))

            print()
            for j in range(len(img_collects )):
                hp = heatmaps_scores[j]
                imgi = img_collects[j]
                imgm = img_masks_collects [j]
                hp = (hp - heatmaps_scores.min()) / (heatmaps_scores.max() - heatmaps_scores.min())
                #hp = (hp - hp.min()) / (hp.max() - hp.min())
                # hp[hp>=0.35] = 1.0
                # hp[hp>=0.5] = 1.0
              
                imgi = np.uint8(imgi.permute(1,2,0).numpy()[:,:,[2,1,0]])
                imgm = np.uint8(imgm.permute(1,2,0).numpy()[:,:,[2,1,0]]).mean(axis=-1)
                imgm = np.int32((imgm>=255))
                threshold = get_threshold(imgm,hp.copy())
                hp[hp>threshold] = 1.0
                # hp[hp>=0.5] = 1.0
                # print('Pixel-level AUROC')
               

               
                # print('pixel ROCAUC: %.3f | best: %.3f'% ( per_pixel_rocauc, best_pxl_roc))
                # print('pixel PROAUC: %.3f | best: %.3f'% ( per_pixel_proauc, best_pxl_pro))
                hp *= 255.0
                hp = cv2.applyColorMap(np.uint8(hp), cv2.COLORMAP_JET)

                visual_map = (hp / 255.0 + imgi / 255.0) * 255.0
                visual_map = (visual_map / visual_map.max()) * 255.0
                
                # n =  p.name.split('.')[0]+'_o_' +save_names[i] + '.jpg'
                # anname = p.name.split('.')[0]+'_an_' +save_names[i] + '.jpg'
                save_an_path = str(save_dir / path_all[j])
                save_img_pah = str(save_dir /path_ori[j])
                print('save ',save_an_path)
                print('save ', save_img_pah)
                cv2.imwrite(save_an_path , np.uint8(visual_map))
                cv2.imwrite(save_img_pah, np.uint8(imgi))
            img_collects  = None
            path_all = []
            path_ori = []
            heatmaps_scores = []
        print(i,'done')
        # for j in range(len(img_ins_all )):
        #     hp = heatmap_all[j]
        #     imgi = img_ins_all[j]
        #     hp = (hp - heatmap_all.min()) / (heatmap_all.max() - heatmap_all.min())
        #     #hp = (hp - hp.min()) / (hp.max() - hp.min())
        #     # hp[hp>=0.75] = 1.0
        #     #hp[hp>=0.5] = 1.0
        #     hp *= 255.0
        #     imgi = np.uint8(imgi.permute(1,2,0).numpy()[:,:,[2,1,0]])
        #     hp = cv2.applyColorMap(np.uint8(hp), cv2.COLORMAP_JET)
        #     visual_map = (hp / 255.0 + imgi / 255.0) * 255.0
        #     visual_map = (visual_map / visual_map.max()) * 255.0
        #     # n =  p.name.split('.')[0]+'_o_' +save_names[i] + '.jpg'
        #     # anname = p.name.split('.')[0]+'_an_' +save_names[i] + '.jpg'
        #     save_an_path = str(save_dir / path_all[j])
        #     save_img_pah = str(save_dir /path_ori[j])
        #     print('save ',save_an_path)
        #     print('save ', save_img_pah)
        #     cv2.imwrite(save_an_path , np.uint8(visual_map))
        #     cv2.imwrite(save_img_pah, np.uint8(imgi))
    # for j in range(len(img_collects )):
    #         hp = heatmaps_scores[j]
    #         imgi = img_collects[j]
    #         hp = (hp - heatmaps_scores.min()) / (heatmaps_scores.max() - heatmaps_scores.min())
    #         #hp = (hp - hp.min()) / (hp.max() - hp.min())
    #         # hp[hp>=0.35] = 1.0
    #         # hp[hp>=0.5] = 1.0
    #         hp *= 255.0
    #         imgi = np.uint8(imgi.permute(1,2,0).numpy()[:,:,[2,1,0]])
    #         hp = cv2.applyColorMap(np.uint8(hp), cv2.COLORMAP_JET)
    #         visual_map = (hp / 255.0 + imgi / 255.0) * 255.0
    #         visual_map = (visual_map / visual_map.max()) * 255.0
    #         # n =  p.name.split('.')[0]+'_o_' +save_names[i] + '.jpg'
    #         # anname = p.name.split('.')[0]+'_an_' +save_names[i] + '.jpg'
    #         save_an_path = str(save_dir / path_all[j])
    #         save_img_pah = str(save_dir /path_ori[j])
    #         print('save ',save_an_path)
    #         print('save ', save_img_pah)
    #         cv2.imwrite(save_an_path , np.uint8(visual_map))
    #         cv2.imwrite(save_img_pah, np.uint8(imgi))
    print('done')

def get_score(im,model,path,device):
    im = torch.from_numpy(im).to(device)
    im = im.float() 
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    detectbox = torch.tensor([[0.30900,0.37937,0.37179,0.42593]])
    heatmap,img_ins,save_names = model(im,detectbox,0)
    # heatmap,img_ins,save_names = model(im,None)
    if heatmap ==None:
        return None,None,None,None
    # heatmap = heatmap.numpy()
    heatmap *= 255.0
    p = Path(path)
    p = p.name
    return heatmap,img_ins,save_names, p

def get_mask(im,device):
    
    im = torch.from_numpy(im).to(device)
    im = im.float() 
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    detectbox = torch.tensor([[0.30900,0.37937,0.37179,0.42593]])
    detectbox *= torch.tensor([640,640,640,640])
    
    detectbox = detectbox.to(device)
    boxes  = detectbox[:,:4]
    wh =  boxes[:,2:] - boxes[:,:2]
    cxcy = boxes[:,:2] + wh/2
    newboxes = torch.cat([cxcy, 1.20*wh],dim=-1)
    newboxes[:,:2] -= newboxes[:,2:]/2
    newboxes[:, 2:] += newboxes[:,:2]
    newboxes = newboxes.clamp(max=640-1,min=0.0)
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


def main(opt):
    run(**vars(opt))

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


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

