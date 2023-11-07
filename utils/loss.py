# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.device = device
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=self.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.unsafe_chunk(4, dim=1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

class ComputeDetectANloss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = de_parallel(model.model_t).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index

        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.device = device
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets,epoach_ratio, expand_ratio=1.5, type='train'):  # predictions, targets

        preds, ftlist, fslist = p
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        lan = torch.zeros(1, device=self.device) # anormaly feature loss
        ldis = torch.zeros(1, device=self.device)  # anormaly feature loss
        lcos = torch.zeros(1, device=self.device)  # anormaly feature loss
        tcls, tbox, indices, anchors = self.build_targets(preds, targets)  # targets
        # epoach_ratio = 0.8 if epoach_ratio> 0.8 else epoach_ratio
        # Losses
        anmaps = []
        for i, (pi,ft,fs) in enumerate(zip(preds,ftlist,fslist)):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                expbox = pbox.detach().clone()
                expbox[:, 0] += gi
                expbox[:, 1] += gj
                expbox[:,2:] *= expand_ratio
                expbox = self.xyxy2ltrb(expbox)
                expbox[:, :2] = expbox[:, :2].floor()
                expbox[:, 2:] = expbox[:, 2:].ceil()
                expbox = expbox.clamp(min=0)
                # if type == 'test':
                #     anmaps.append(anmap)

                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss
                #pobj = pi[b,a,gj,gi][:,4].clone().detach().sigmoid()
                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                pobj = iou.clone()

                pobj[pobj >= 0.8] = 1.0
                pobj[pobj<=0.15] = 0.0
                anloss,disloss,cosloss = self.get_anfeatureloss(expbox,pobj, ft, fs, b, type=type)
                lan += anloss
                ldis += disloss
                lcos += cosloss
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box'] *3
        lobj *= self.hyp['obj'] *3
        lcls *= self.hyp['cls'] *3
        lan *= self.hyp['an']
        ldis *= self.hyp['an']
        lcos *= self.hyp['an']
        bs = tobj.shape[0]  # batch size
        detectloss = (lbox + lobj + lcls) * bs
        featureanloss = lan * bs
        # detectloss = (lbox + lobj + lcls) * bs * (1-epoach_ratio)
        # featureanloss = 0.02 * lan * bs * epoach_ratio
        return detectloss + featureanloss, torch.cat((lbox, lobj, lcls, lan,ldis,lcos)).detach()


    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=self.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.unsafe_chunk(4, dim=1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

    def xyxy2ltrb(self,box):
        box[:, :2] -= box[:, 2:]/2
        box[:, 2:] += box[:, :2]
        return box

    def ltrb2xyxy(self,box):
        box[:, 2:] -= box[:, :2]
        box[:, :2] += box[:, 2:]/2
        return box

    def get_anfeatureloss(self,boxes, objscores,ftlist, fslist,batchid,type='train'):
        nb = len(ftlist)
        # disout = []
        # cosout = []
        # ssout = []
        tsloss = 0
        total_dis = 0
        total_cos = 0
        for i in range(nb):
            box = boxes[batchid == i].long()
            obj = objscores[batchid == i]
            ft = ftlist[i]
            fs = fslist[i]
            assert ft.shape == fs.shape

            fta = [ft[:,box[j,1]:box[j,3]+1,box[j,0]:box[j,2]+1] for j in range(box.shape[0])]
            fsa = [fs[:, box[j, 1]:box[j, 3] + 1, box[j, 0]:box[j, 2] + 1] for j in range(box.shape[0])]
            if len(fta) == 0 or len(box) == 0 or len(fsa) == 0:
                loss = 0
                disloss = 0
                cosloss = 0
                # disout.append([])
                # cosout.append([])
                # ssout.append([])

            else:
                lossouts = [self.featureloss(fta[j],fsa[j],obj[j]) for j in range(len(fta))]
                loss = torch.stack([an[0] for an in lossouts], dim=0).mean()
                disloss = torch.stack([an[1][0] for an in lossouts], dim=0).mean()
                cosloss = torch.stack([an[1][1] for an in lossouts], dim=0).mean()
                # loss = torch.stack([an[0] for an in lossouts],dim=0).mean()
                # if type=='test':
                #     dismaps = [an[1][0] for an in lossouts]
                #     cosmaps = [an[1][1] for an in lossouts]
                #     ssimmaps = [an[1][2] for an in lossouts]
                #     disout.append(dismaps)

            tsloss += loss
            total_cos += cosloss
            total_dis += disloss
        return tsloss/nb, total_dis/nb ,total_cos/nb

    def make_proposalmask(self,boxes,featureshape):
        mask = torch.zeros([featureshape[1],featureshape[2]],device= boxes.device)*-1
        for box in boxes:
            mask[box[1]:box[3] + 1, box[0]:box[ 2] + 1] = 1.0
        return mask>0

    def featureloss(self, ft, fs,obj):
        assert ft.shape == fs.shape
        if len(ft) ==0:
            return 0,(None,None,None)
        _,h,w =  ft.shape
        # dis = abs
        # ft_std = (ft - ft.mean())/ft.std()
        # fs_std = (fs - fs.mean())/fs.std()
        # min_val = min(ft.min(),fs.min())
        # ft_std -= min_val
        # fs_std -= min_val
        ft_norm = F.normalize(ft, p=2,dim=0 )
        fs_norm = F.normalize(fs, p=2, dim=0)
        dissim = (0.5 * (ft_norm - fs_norm) **2).sum(dim=0)
        # dissim =
        cossim = F.cosine_similarity((ft).unsqueeze(dim=0),
                                     (fs).unsqueeze(dim=0))
        # cossim = F.cosine_similarity((ft.flatten(1)).unsqueeze(dim=0), (fs_std.flatten(1)).unsqueeze(dim=0), dim=-1)
        cosloss = (1-cossim).mean()
        #disloss = dissim.mean() * obj
        disloss = F.l1_loss(dissim, (1 - obj) * torch.ones(dissim.shape, device=dissim.device))
        outloss = (disloss + cosloss)
        anmap = dissim
        cosmap = cossim.squeeze(dim=0)
        return outloss, (disloss,  cosloss)
        # ft_exp = ft.exp()
        # fs_exp = fs.exp()
        # ft_norm = F.normalize(ft_exp, p=2, dim=0)
        # fs_norm = F.normalize(fs_exp, p=2, dim=0)
        # ft_c = ft.flatten(1)
        # fs_c = fs.flatten(1)
        # ft_cnorm = F.normalize(ft_c, p=2, dim=1)
        # fs_cnorm = F.normalize(fs_c, p=2, dim=1)
        # ft_c = ft_c - ft_c.min(dim=1)[0].unsqueeze(-1).repeat([1, h * w])
        # fs_c = fs_c - fs_c.min(dim=1)[0].unsqueeze(-1).repeat([1, h * w])


        # cossimc = F.cosine_similarity((ft_cnorm ).unsqueeze(dim=0),
        #                              (fs_cnorm ).unsqueeze(dim=0))
        # cossim = F.cosine_similarity((ft_norm - ft_norm.mean(dim=0)).unsqueeze(dim=0),
        #                              (fs_norm - fs_norm.mean(dim=0)).unsqueeze(dim=0))
        # cossim = F.cosine_similarity((ft - ft.mean(dim=0)).unsqueeze(dim=0),
        #                              (fs - fs.mean(dim=0)).unsqueeze(dim=0))
        # ssim = self.ssim_loss(ft_norm,fs_norm)
        #ssim = self.ssim_loss(ft, fs)
        # ssim = self.ssim_loss(ft_c.T, fs_c.T).mean()
        # ssimc = self.ssim_loss(ft_c.T, fs_c.T)
        # dissim = (0.5 * (ft_norm - fs_norm) ** 2).sum(dim=0, keepdim=True)
        #dissimc = (0.5 * (ft_cnorm - fs_cnorm) ** 2).sum(dim=1, keepdim=True)

        # ssimcloss = F.l1_loss(ssimc, obj * torch.ones(ssimc.shape, device=ssim.device))
        # coscloss = F.l1_loss(cossimc, obj * torch.ones(cossimc.shape, device=cossim.device))
        # discloss = F.l1_loss(dissimc, (1 - obj) * torch.ones(dissimc.shape, device=dissim.device))
        # ssimcloss = (1 - ssimc).mean()
        # coscloss = (1 - cossimc).mean()
        # discloss = (dissimc).mean()

       # simloss = F.l1_loss(ssim, obj * torch.ones(ssim.shape,device=ssim.device))
       #  cosloss = F.l1_loss(cossim, obj * torch.ones(cossim.shape,device=cossim.device))
       #  disloss = F.l1_loss(dissim, (1-obj)*torch.ones(dissim.shape,device=dissim.device))
        # ssimloss += ssimcloss
        # cosloss += coscloss
        # disloss += discloss
        # ssimloss = (1-ssim).mean()
        # cosloss = (1 - cossim).mean()
        # disloss = dissim.mean()
        #dissim = F.mse_loss(fs_norm, ft_norm, reduction='none').sum(dim=0,keepdim=True)

        #outloss =  cosloss + ssimloss
        # outloss = (disloss + cosloss)
        # anmap = dissim
        # cosmap = cossim.squeeze(dim=0)
        # outloss = outloss * obj
        #ssimmap = ssim
        #return outloss, ( cosmap, ssimmap)
        #
        # return outloss, (anmap,cosmap)
        # return outloss,(anmap,cosmap)

    def ssim_loss(self,x,y):

        mulx = (x ).mean(dim=0)
        muly = (y ).mean(dim=0)
        mulxy = mulx* muly
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        sigmax = (x ).std(dim=0)
        sigmay = (y ).std(dim=0)
        sigmaxy = sigmax * sigmay
        ssim = ((2* mulxy+c1)*(2*sigmaxy+c2))/((mulx**2 +muly**2 + c1)*(sigmax**2+sigmay**2 +c2))
        return ssim

def get_dissim( ft, fs):
    assert ft.shape == fs.shape
    ft_norm = ft.exp()
    fs_norm = fs.exp()
    ft_norm = F.normalize(ft_norm, p=2, dim=0)
    fs_norm = F.normalize(fs_norm, p=2, dim=0)
    dissim = (0.5 * (ft_norm - fs_norm) ** 2).sum(dim=0)
    # ft_norm = ft - ft.min(dim=0)[0].expand_as(ft)
    # fs_norm = fs - fs.min(dim=0)[0].expand_as(fs)
    # ft_norm = F.normalize(ft_norm, p=2, dim=0)
    # fs_norm = F.normalize(fs_norm, p=2, dim=0)
    # dissim = F.mse_loss(fs_norm, ft_norm, reduction='none').sum(dim=0)
    return dissim

def get_cossim(ft,fs):
    assert ft.shape == fs.shape
    _, h, w = ft.shape
    ft_norm = ft
    fs_norm = fs
    # ft_norm = ft.exp()
    # fs_norm = fs.exp()
    # ft_norm = F.normalize(ft_norm, p=2, dim=0)
    # fs_norm = F.normalize(fs_norm, p=2, dim=0)
    # ft_norm = ft - ft.min(dim=0)[0].expand_as(ft)
    # fs_norm = fs - fs.min(dim=0)[0].expand_as(fs)
    # ft_norm = F.normalize(ft_norm, p=2, dim=0)
    # fs_norm = F.normalize(fs_norm, p=2, dim=0)
    # ft_norm = ft
    # fs_norm = fs
    cossim = F.cosine_similarity((ft_norm ).unsqueeze(dim=0),
                                 (fs_norm ).unsqueeze(dim=0))
    return cossim

def get_ssim( x, y):
    mulx = (x).mean(dim=0)
    muly = (y).mean(dim=0)
    mulxy = mulx * muly
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    sigmax = (x).std(dim=0)
    sigmay = (y).std(dim=0)
    sigmaxy = sigmax * sigmay
    ssim = ((2 * mulxy + c1) * (2 * sigmaxy + c2)) / ((mulx ** 2 + muly ** 2 + c1) * (sigmax ** 2 + sigmay ** 2 + c2))
    return ssim


class ComputeAN_Loss:
    def __init__(self, model, autobalance=False):
        device = next(model.model_s.parameters()).device
        #h = model.hyp  # hyperparameters
        self.device = device
        self.model = model

    def __call__(self, ft_outlist, fs_outlist, detect,reduce='mean', type='train'):

        assert len(ft_outlist) == len(fs_outlist)
        loss, anmaps = self.train(ft_outlist, fs_outlist, detect, reduce='sum',type= type)
        return loss, anmaps


    def test(self,ftout_batchlist, fsout_batchlist,detect,img_shape):
        self.img_shape = img_shape
        #assert len(ft_outlist) == len(fs_outlist)
        nb = len(detect)
        anmaps_batch = []
        for i in range(nb):
            anmaps_batch.append(self.cal_insanmaps_onebatch(ftout_batchlist[i],fsout_batchlist[i],detect[i],type='test'))
        return  anmaps_batch
        # ni = len(fs_outlist)
        # anomaly_instance_maps = []
        # for i in range(ni):
        #     anomaly_instance_maps.append(self.cal_instance_anmap(ft_outlist[i], fs_outlist[i],detect[i]))
        # return anomaly_instance_maps

    def train(self, ft_outlist, fs_outlist, detect,reduce='mean',type= 'train'):
        assert len(ft_outlist) == len(fs_outlist)
        nb = len(fs_outlist)
        # loss_batch = []
        anmaps_batch = []
        anloss = 0
        for i in range(nb):
            ft_ins = ft_outlist[i]
            fs_ins = fs_outlist[i]
            assert len(ft_ins) == len(fs_ins)
            ni = len(fs_ins)
            if ni == 0:
                continue
            if type =='test':
                anmaps = self.cal_insanmaps_onebatch(ft_outlist[i], fs_outlist[i], detect[i],type=type)
                anmaps_batch.append(anmaps)
            elif type == 'train':
                loss = self.cal_insanmaps_onebatch(ft_outlist[i], fs_outlist[i], detect[i],type=type)
                anloss += loss
            elif type == 'pic':
                loss, anmaps = self.cal_insanmaps_onebatch(ft_outlist[i], fs_outlist[i], detect[i],type=type)
                anmaps_batch.append(anmaps)
                anloss += loss
            # loss, anmaps = self.cal_insanmaps_onebatch(ft_outlist[i], fs_outlist[i], detect[i],type='train')
            # anloss += loss


        if reduce =='mean':
            anloss /= nb

        return anloss, anmaps_batch


    def cal_frame_anloss(self,ft_framelist,fs_framelist):
        assert len(ft_framelist) ==len(fs_framelist)
        nf = len(fs_framelist)
        t_loss = []
        for i in range(nf):
            loss = self.cal_instance_loss(ft_framelist[i],fs_framelist[i])
            if len(loss) ==0:
                continue
            t_loss.append(loss)

        return torch.stack(t_loss,dim=1).mean(dim=1)

    def cal_instance_loss(self,ft_list, fs_list,reduce ='sum'):
        total_loss = []
        ni = len(fs_list)
        for i in range(ni):
            total_loss.append( self.cal_insfeature_loss(ft_list[i],fs_list[i]))
        if reduce =='mean':
            loss = torch.stack(total_loss, dim=0).mean()
        else:
            loss = torch.stack(total_loss,dim=0).sum()
        return loss

    def cal_insfeature_loss(self, ft_lvls, fs_lvls):
        assert len(ft_lvls) == len(fs_lvls)
        nl = len(fs_lvls)
        total_loss = 0
        losses = []
        for i in range(nl):
            ft = ft_lvls[i]
            fs = fs_lvls[i]
            _, h, w = fs.shape
            fs_norm = F.normalize(fs, p=2)
            ft_norm = F.normalize(ft, p=2)
            f_loss = 0.5 * (ft_norm - fs_norm) ** 2
            losses.append(f_loss)
            f_loss = f_loss.sum() / (h * w)
            total_loss += f_loss

        total_loss /= nl

        return total_loss


    def cal_frame_anmap(self,ft_framelist,fs_framelist):
        assert len(ft_framelist) == len(fs_framelist)
        nf = len(fs_framelist)
        for i in range(nf):
            self.cal_anomaly_instance_map(ft_framelist[i], fs_framelist[i])

    def cal_instance_anmap(self, ft_list, fs_list, box):
        anomaly_map = 1
        anomaly_map_sum = 0
        assert len(ft_list) == len(fs_list)
        nl = len(fs_list)
        bw, bh = box[2:4] - box[:2]
        bw = bw.ceil().int()
        bh = bh.ceil().int()

        for i in range(nl):

            fs = fs_list[i]
            ft = ft_list[i]
            _, h, w = fs.shape
            fs_norm = F.normalize(fs, p=2)
            ft_norm = F.normalize(ft, p=2)
            a_map = (0.5 * (ft_norm - fs_norm) ** 2) / (h * w)
            a_map = 1 - F.cosine_similarity(fs_norm, ft_norm)
            #a_map = a_map.sum(0, keepdim=True)
            a_map = a_map.unsqueeze(dim=0)
            a_map = F.interpolate(a_map, size=[bh,bw], mode='bilinear', align_corners=False)
            a_map = a_map.squeeze(dim=0).squeeze(dim=0)
            #anomaly_map += a_map
            anomaly_map *= a_map
            anomaly_map_sum += a_map
        anomaly_map = anomaly_map.squeeze().cpu().numpy()
        anomaly_map = gaussian_filter(anomaly_map, sigma=4)
        return anomaly_map

    def cal_insanmaps_onebatch(self,ft_list, fs_list, boxes,reduce='mul',type='test'):
        assert len(ft_list) == len(fs_list)
        ni = len(boxes)
        anmaps = []

        anloss = 0
        for i in range(ni):
            if type == 'pic':
                anmap, a_loss = self.cal_instance_anmaps(ft_list[i], fs_list[i], boxes[i], reduce=reduce, type=type)
                anloss += a_loss
                anmaps.append(anmap)
            elif type == 'test':
                anmap = self.cal_instance_anmaps(ft_list[i], fs_list[i], boxes[i], reduce=reduce, type=type)
                anmaps.append(anmap)
            else:
                a_loss = self.cal_instance_anmaps(ft_list[i], fs_list[i], boxes[i], reduce=reduce, type=type)
                anloss += a_loss

        if type == 'test':
            return anmaps
        if type == 'pic':
            return anloss/ni, anmaps
        else:
            anloss = anloss/ni
            return anloss

    def cal_instance_anmaps(self,ft_features,fs_features, box,reduce='sum',type='train'):
        assert len(ft_features) == len(fs_features)
        nl = len(ft_features)
        outloss = 0
        anomaly_map = 0.0
        ancos_map = 0.0

        bw, bh = box[2:4].floor() - box[:2].floor()
        bw = bw.ceil().int()
        bh = bh.ceil().int()
        out_size = [bh, bw]


        for i in range(nl):
            ft = ft_features[i]
            fs = fs_features[i]
            # ft = ft -ft.min()
            # fs = fs - fs.min()
            c, h, w = fs.shape
            # ft_norm = self.standlization(ft,dim=0)
            # fs_norm = self.standlization(fs, dim=0)
            ft_norm = ft - ft.min(dim=0)[0].expand_as(ft)
            fs_norm = fs - fs.min(dim=0)[0].expand_as(fs)
            ft_norm = F.normalize(ft_norm, p=2, dim=0)
            fs_norm = F.normalize(fs_norm, p=2, dim=0)

            ft_c = ft.flatten(1)
            fs_c = fs.flatten(1)
            # ft_c = (ft.flatten(1) - ft.flatten(1).mean(dim=-1).unsqueeze(-1).repeat([1,h*w]))/ft.flatten(1).std(dim=-1).unsqueeze(-1).repeat([1,h*w])
            # fs_c = (fs.flatten(1) - fs.flatten(1).mean(dim=-1).clone().detach().unsqueeze(-1).repeat([1,h*w]))/fs.flatten(1).std(dim=-1).unsqueeze(-1).repeat([1,h*w])
            ft_c = ft_c - ft_c.min(dim=1)[0].unsqueeze(-1).repeat([1,h*w])
            fs_c = fs_c - fs_c.min(dim=1)[0].unsqueeze(-1).repeat([1,h*w])

            ft_cnorm = F.normalize(ft_c, p=2,dim=1)
            fs_cnorm = F.normalize(fs_c, p=2,dim=1)
            cossim = F.cosine_similarity((ft_norm - ft_norm.mean(dim=0)).unsqueeze(dim=0),
                                         (fs_norm - fs_norm.mean(dim=0)).unsqueeze(dim=0))
            # ccossim = F.cosine_similarity((ft_cnorm - ft_cnorm.mean(dim=1).unsqueeze(dim=1).repeat([1,h*w])).unsqueeze(dim=0).unsqueeze(dim=0),
            #                               (fs_cnorm - fs_cnorm.mean(dim=1).unsqueeze(dim=1).repeat([1,h*w])).unsqueeze(dim=0).unsqueeze(dim=0),dim=-1)
            #ccossim = ccossim.squeeze(dim=0).squeeze(dim=0)
            # cosloss = (1 - cossim).sum() + (1 - ccossim).sum()
            cosloss = (1 - cossim).sum()
            dissim = F.mse_loss(fs_norm,ft_norm,reduction='none').sum(dim=0)
            #channelsim = F.mse_loss(fs_cnorm,ft_cnorm,reduction='none').sum(dim=1)
            #disloss = dissim.sum() + channelsim.sum()
            disloss = dissim.sum()
            outloss += disloss + cosloss
            # outloss += anloss + cosloss * distance.sum(dim=0).mean().clone().detach()
            if type == 'pic' or 'test':

                # an_map = dissim.expand_as(fs) + channelsim.unsqueeze(dim=-1).repeat(1,h*w).view(c,h,w)
                # an_map /= 2.0
                an_map = dissim.expand_as(fs)
                an_map = an_map.mean(dim=0)
                # cosmap = (1-cossim[0]).expand_as(fs) + (1-ccossim).unsqueeze(dim=-1).repeat(1,h*w).view(c,h,w)
                # cosmap /= 2.0
                cosmap = (1 - cossim[0]).expand_as(fs)
                cosmap = cosmap.mean(dim=0)
                an_map = an_map.unsqueeze(dim=0).unsqueeze(dim=0)

                an_map = F.interpolate(an_map, size=out_size, mode='bilinear', align_corners=False)
                an_map = an_map.squeeze(dim=0).squeeze(dim=0)

                cosmap = cosmap.unsqueeze(dim=0).unsqueeze(dim=0)
                cosmap = F.interpolate(cosmap, size=out_size, mode='bilinear', align_corners=False)
                cosmap = cosmap.squeeze(dim=0).squeeze(dim=0)
                anomaly_map += an_map
                ancos_map += cosmap

        ancos_map /= nl
        anomaly_map /= nl
        outloss /= nl


        if type == 'test':
            return (anomaly_map.cpu().numpy(),ancos_map.cpu().numpy())
        if type == 'train':
            return outloss
        if type == 'pic':
            return (anomaly_map,ancos_map), outloss






    def cal_loss(self,fs_list, ft_list):
        t_loss = 0
        N = len(fs_list)
        for i in range(N):
            fs = fs_list[i]
            ft = ft_list[i]
            _, _, h, w = fs.shape
            fs_norm = F.normalize(fs, p=2)
            ft_norm = F.normalize(ft, p=2)
            f_loss = 0.5 * (ft_norm - fs_norm) ** 2
            f_loss = f_loss.sum() / (h * w)
            t_loss += f_loss

        return t_loss / N

    def cal_anomaly_maps(self,fs_list, ft_list, out_size):
        anomaly_map = 0
        for i in range(len(ft_list)):
            fs = fs_list[i]
            ft = ft_list[i]
            fs_norm = F.normalize(fs, p=2)
            ft_norm = F.normalize(ft, p=2)
            _, _, h, w = fs.shape
            a_map = (0.5 * (ft_norm - fs_norm) ** 2) / (h * w)
            a_map = a_map.sum(1, keepdim=True)
            a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=False)
            anomaly_map += a_map
        anomaly_map = anomaly_map.squeeze().cpu().numpy()
        for i in range(anomaly_map.shape[0]):
            anomaly_map[i] = gaussian_filter(anomaly_map[i], sigma=4)

        return anomaly_map

    def standlization(self,input,dim=0):
        std = input.std(dim=dim)
        mean = input.mean(dim=dim)
        return (input - mean.expand_as(input))/std.expand_as(input)

    def normalize(self,input):
        norm = input.clone()
        for i in range(input.shape[0]):
            norm[i] = input[i]/ input[i].norm(p=1)
        return norm

    def cal_reldis(self,distance):
        c = distance.shape[0]

        outs = [((distance - distance[i].clone().detach().expand_as(distance)).abs().sum(dim=0)/(c-1)).sum() for i in range(c)]
        disloss = torch.stack(outs,dim=0).sum()
        # for i in range(c):
        #     dis = (distance - distance[i].expand_as(distance)).abs()
        #     disloss += (dis.sum(dim=0)/(c-1)).sum()
        return disloss

    def liner_normalize(self,input):
        norms = []
        for i in range(input.shape[0]):
            min = input[i].min().detach()
            max = input[i].max().detach()
            max -= min
            norm = input[i]
            norms.append()

class ComputeAN_Loss2:
    def __init__(self, model, autobalance=False):
        device = next(model.model_s.parameters()).device
        #h = model.hyp  # hyperparameters
        self.device = device
        self.model = model

    def __call__(self, ft_outlist, fs_outlist, mask_list,detect,reduce='mean'):

        assert len(ft_outlist) == len(fs_outlist)
        loss = self.train(ft_outlist, fs_outlist,mask_list,detect, reduce='sum')
        return loss


    def test(self,ftout_batchlist, fsout_batchlist,detect,img_shape):
        self.img_shape = img_shape
        #assert len(ft_outlist) == len(fs_outlist)
        nb = len(detect)
        anmaps_batch = []
        for i in range(nb):

            anmaps_batch.append(self.cal_insanmaps_onebatch(ftout_batchlist[i],fsout_batchlist[i],detect[i]))
        return  anmaps_batch
        # ni = len(fs_outlist)
        # anomaly_instance_maps = []
        # for i in range(ni):
        #     anomaly_instance_maps.append(self.cal_instance_anmap(ft_outlist[i], fs_outlist[i],detect[i]))
        # return anomaly_instance_maps

    def train(self, ft_outlist, fs_outlist, mask_list, detect,reduce='sum',outsize =(960,960)):
        assert len(ft_outlist) == len(fs_outlist)
        nl = len(fs_outlist)
        anloss = 0
        for i in range(nl):
            ft_batch = ft_outlist[i]
            fs_batch = fs_outlist[i]
            mask = mask_list[i]
            mask = mask.cpu().long()
            ft_norm = F.normalize(ft_batch, p=2)
            fs_norm = F.normalize(fs_batch, p=2)

            b,_,h,w = ft_batch.shape
            if reduce == 'sum':
                loss = 0.5 * (ft_norm - fs_norm) ** 2
                loss = loss.sum(dim=1)
                loss = loss.cpu()
                loss[~mask]  *= 0
                a_map = loss.detach()
                a_map = a_map.squeeze(dim=1)
                a_map = F.interpolate(loss[0], size=out_size(), mode='bilinear', align_corners=False)
                # a_map = a_map.squeeze(dim=0)
                # a_map = a_map.unsqueeze(dim=0).unsqueeze(dim=0)
                # a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=False)
                # a_map = a_map.squeeze(dim=0)
                anloss += loss
            print()
            #mask = mask_list[i]







