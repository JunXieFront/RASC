# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES
from .cross_entropy_loss import CrossEntropyLoss


class PPC(nn.Module):
    def __init__(self, ignore_index):
        super(PPC, self).__init__()

        self.ignore_index = ignore_index

    def forward(self, contrast_logits, contrast_target):
        loss_ppc = F.cross_entropy(contrast_logits, contrast_target.long(), ignore_index=self.ignore_index)

        return loss_ppc


class PPD(nn.Module):
    def __init__(self, ignore_index):
        super(PPD, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, contrast_logits, contrast_target):
        contrast_logits = contrast_logits[contrast_target != self.ignore_index, :]
        contrast_target = contrast_target[contrast_target != self.ignore_index]

        logits = torch.gather(contrast_logits, 1, contrast_target[:, None].long())
        loss_ppd = (1 - logits).pow(2).mean()

        return loss_ppd


@LOSSES.register_module()
class PixelPrototypeCELoss(nn.Module):
    def __init__(self,
                 loss_ppc_weight,
                 loss_ppd_weight,
                 ignore_index):
        super(PixelPrototypeCELoss, self).__init__()

        self.loss_ppc_weight = loss_ppc_weight
        self.loss_ppd_weight = loss_ppd_weight
        self.seg_criterion = CrossEntropyLoss()

        self.ppc_criterion = PPC(ignore_index)
        self.ppd_criterion = PPD(ignore_index)

    def forward(self,
                out_seg,
                contrast_logits,
                contrast_target,
                target,
                seg_weight=None):
        """
        Args:
            out_seg: [2, 19, 128, 128]
            contrast_logits: [2 * 128 * 128, 190]
            contrast_target: [2 * 128 * 128,]
            target: [2, 512, 512]
        Returns:

        """
        if target.size(1) == 1:
            target = target.squeeze(1)

        loss_ppc = self.ppc_criterion(contrast_logits, contrast_target)
        loss_ppd = self.ppd_criterion(contrast_logits, contrast_target)
        loss_seg = self.seg_criterion(out_seg, target, seg_weight, ignore_index=255)
        loss = loss_seg + self.loss_ppc_weight * loss_ppc + self.loss_ppd_weight * loss_ppd
        return loss


