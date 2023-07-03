"""
Fewshot Semantic Segmentation
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from .vgg import Encoder


def Weighted_GAP(supp_feat, mask):
    mask = F.interpolate(mask, size=(supp_feat.size(2), supp_feat.size(3)), mode='bilinear', align_corners=True)
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area  
    return supp_feat


def TAM(target_feat, ref_label, ref_pre, ref_feat):
    shot = ref_label.size(1)
    ref_label = F.interpolate(ref_label.float(), size=(ref_feat.size(3), ref_feat.size(4)), mode='bilinear', align_corners=True)
    # trimap = torch.where(ref_label != ref_pre, ref_label * 0.5, ref_pre)
    bg_mask = torch.where(ref_label==1, 0, 1).float()
    fgs_prototype = Weighted_GAP(ref_feat[:, 0, :, :], ref_pre[:, 0, :, :].unsqueeze(1))
    bgs_prototype = Weighted_GAP(ref_feat[:, 0, :, :], bg_mask[:, 0, :, :].unsqueeze(1))
    for i in range(1, shot):
        fgs_prototype += Weighted_GAP(ref_feat[:, 0, :, :], ref_pre[:, i, :, :].unsqueeze(1))
        bgs_prototype += Weighted_GAP(ref_feat[:, 0, :, :], bg_mask[:, i, :, :].unsqueeze(1))
    fgs_prototype /= shot
    bgs_prototype /= shot
    fg_dist = F.cosine_similarity(target_feat, fgs_prototype).unsqueeze(1)
    bg_dist = F.cosine_similarity(target_feat, bgs_prototype).unsqueeze(1)
    result = torch.cat([bg_dist, fg_dist], dim=1)
    return result


class FewShotSeg(nn.Module):
    """
    Fewshot Segmentation model

    Args:
        in_channels:
            number of input channels
        pretrained_path:
            path of the model for initialization
        cfg:
            model configurations
    """
    def __init__(self, in_channels=3, pretrained_path=None, cfg=None):
        super().__init__()
        self.pretrained_path = pretrained_path
        self.config = cfg or {'align': False}

        # Encoder
        self.encoder = nn.Sequential(OrderedDict([
            ('backbone', Encoder(in_channels, self.pretrained_path)),]))


    def forward(self, s_x, s_y, q_x, q_y):
        """
        Args:
            s_x: support images [B X K X C X H X W] tensors

            s_y: support annotation [K X 25 X 4] tensors

            q_x: query images [B X C X H X W] tensors

            q_y: query mask [B X H X W] tensors
        """
        shape = s_x.shape
        n_shots = shape[1]
        batch_size = shape[0]
        img_size = (shape[3], shape[4])

        ###### Extract features ######
        supp_img = torch.cat([s_x[:,shot,:,:,:] for shot in range(n_shots)])
        
        supp_fts = self.encoder(supp_img)
        fts_size = supp_fts.shape[-2:]
        supp_fts = supp_fts[:n_shots * batch_size].view(
            batch_size, n_shots,  -1, *fts_size)

        mask = (s_y[:,0,:,:] == 1).float().unsqueeze(1)
        fgs_prototype = Weighted_GAP(supp_fts[:, 0, :, :], mask)
        for i in range(1, n_shots):
            mask = (s_y[:,i,:,:] == 1).float().unsqueeze(1)
            fgs_prototype += Weighted_GAP(supp_fts[:, i, :, :], mask)

        fgs_prototype /= n_shots

        qry_fts = self.encoder(q_x)
        fg_dist = F.cosine_similarity(qry_fts, fgs_prototype)
        pre_mask = ((fg_dist > 0.7)).float().unsqueeze(1)
        
        for  e in range(4):
            if e % 2 == 0:
                _pre_mask = []
                for i in range(n_shots):
                    pre_mask_one = TAM(supp_fts[:, i, :, :], q_y.float().unsqueeze(1), pre_mask, qry_fts.unsqueeze(1))
                    _, pre_mask_one = torch.max(pre_mask_one, 1)
                    _pre_mask.append(pre_mask_one.unsqueeze(1).float())
                pre_mask = torch.cat([mask for mask in _pre_mask], dim=1)
            else:
                pre_mask = TAM(qry_fts, s_y, pre_mask, supp_fts)

        pre_mask = F.interpolate(pre_mask, size=img_size, mode='bilinear', align_corners=True)
        _, pred = torch.max(pre_mask, 1)
        pred = pred.long()
        q_y = q_y.long()
        new_label = torch.where(((pred==0) & (q_y==1)), 255, pred)
        return pre_mask, new_label



        
