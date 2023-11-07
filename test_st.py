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

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5x.pt', help='model path(s)')
    parser.add_argument('--weights_an', nargs='+', type=str, default=ROOT / 'st_199.pt', help='model path(s)')
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
    parser.add_argument('--project', default=ROOT / 'runs/st/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt

@torch.no_grad()
def run(weights=ROOT / 'yolov5x.pt',  # model.pt path(s)
        weights_an = ROOT/'st_299.pt',
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
    device = torch.device('cuda:1')
    model = AN_ST(ckpt=weights_an,detect=True,device=device)
    # model.model_s.eval()
    stride, names = model.stride, model.names
    names = ["car","truck","bus","person","fire","smoke","cone","div","suit","box","moto","hat","ma","bucket"]
    pt = True
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    source_an = "/mnt/ecc5c6c9-7631-4983-9ba0-1ec98729589b/Dataset/ancfa/test/4k925_647"
    source = "/mnt/ecc5c6c9-7631-4983-9ba0-1ec98729589b/Dataset/ancfa/train/4k925_647"
    
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    dataset_an = LoadImages(source_an, img_size=imgsz, stride=stride, auto=pt)
    dataset.__iter__()
    dataset_an.__iter__()
    bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    heatmaps_scores = None
    count = 0
    img_collects = None
    # path_all = []
    # path_ori = []
    for i in range(len(dataset_an)):
    # for i in range(min(len(dataset_an),len(dataset))):
        
        path, im, _, _, _ = dataset.__next__()
        path_an, im_an, _, _, _ =dataset_an.__next__()
        heatmap,img_ins,save_names, p = get_score(im,model,path,device)
        heatmap_an,img_ins_an,save_names_an, p_an = get_score(im_an,model,path_an,device)
        # if heatmap_an == None :
        #     continue
        if heatmap_an == None or heatmap ==None:
            continue

        img_ins_all = img_ins_an
        heatmap_all = heatmap_an
        # img_ins_all = img_ins
        # heatmap_all = heatmap
        path_all = []
        path_ori = []
        # img_ins_all = torch.cat([img_ins,img_ins_an],dim=0)
        # heatmap_all = torch.cat([heatmap,heatmap_an],dim=0)
        heatmap_all = heatmap_all.numpy()
       
        # for n in save_names:
        #     path_all.append(str(i)+'_nan_' +n + '.jpg' )
        #     path_ori.append(str(i)+'_o_' +n + '.jpg')
        for n in save_names_an:
            path_all.append(str(i)+'_an_' +n + '.jpg' )
            path_ori.append(str(i)+'_o_' +n + '.jpg')
        # heatmaps_scores = torch.cat((heatmaps_scores, heatmap_all), dim=0) if heatmaps_scores != None else heatmap_all
        # img_collects = torch.cat((img_collects, img_ins_all ), dim=0) if img_collects != None else img_ins_all 
        for j in range(len(img_ins_all )):
            hp = heatmap_all[j]
            imgi = img_ins_all[j]
            hp = (hp - heatmap_all.min()) / (heatmap_all.max() - heatmap_all.min())
            #hp = (hp - hp.min()) / (hp.max() - hp.min())
            # hp[hp>=0.75] = 1.0
            #hp[hp>=0.5] = 1.0
            hp *= 255.0
            imgi = np.uint8(imgi.permute(1,2,0).numpy()[:,:,[2,1,0]])
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
    # for j in range(len(img_collects )):
    # # for j in range(len(10 )):
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
    heatmap,img_ins,save_names = model(im,None)
    if heatmap ==None:
        return None,None,None,None
    # heatmap = heatmap.numpy()
    heatmap *= 255.0
    p = Path(path)
    p = p.name
    return heatmap,img_ins,save_names, p

def roc_auc_img(gt, score):
    img_roc_auc = roc_auc_score(gt, score)
    
    return img_roc_auc

def roc_auc_pxl(gt, score):
    per_pixel_roc_auc = roc_auc_score(gt.flatten(), score.flatten())

    return per_pixel_roc_auc


def pro_auc_pxl(gt, score):
    gt = np.squeeze(gt, axis=1)

    gt[gt <= 0.5] = 0
    gt[gt > 0.5] = 1
    gt = gt.astype(np.bool)

    max_step = 200
    expect_fpr = 0.3

    max_th = score.max()
    min_th = score.min()
    delta = (max_th - min_th) / max_step

    pros_mean = []
    pros_std = []
    threds = []
    fprs = []

    binary_score_maps = np.zeros_like(score, dtype=np.bool)

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

