import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from .proformer_utils import momentum_update, l2_normalize, distributed_sinkhorn
from mmcv.runner import force_fp32
from ..builder import HEADS, build_loss
from mmseg.ops import resize
from ..losses import accuracy
from .decode_head import BaseDecodeHead


class ProjectionHead(nn.Module):
    def __init__(self, in_channels, proj_dim=256):
        super(ProjectionHead, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, proj_dim, kernel_size=1)
        )

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)


@HEADS.register_module()
class ProFormerHead(BaseDecodeHead):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_classes,
                 num_prototypes,
                 gamma,
                 loss_decode=None):
        super().__init__(
            in_channels,
            out_channels,
            num_classes=num_classes,
            loss_decode=loss_decode,
            align_corners=True
        )
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Dropout2d(0.10)
        )

        self.proj_head = ProjectionHead(in_channels, out_channels)
        self.mask_norm = nn.LayerNorm(num_classes)
        self.feat_norm = nn.LayerNorm(out_channels)
        self.pixel_loss = build_loss(loss_decode)
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.gamma = gamma
        self.prototypes = nn.Parameter(torch.zeros(num_classes, num_prototypes, out_channels),
                                       requires_grad=True)

        nn.init.trunc_normal_(self.prototypes, std=0.02)

    def prototype_learning(self, _c, out_seg, gt_seg, masks):
        """
        _c:[n, d]
        out_seg:[b, k, h, w]
        gt_seg:[n, ]
        mask:[n, m ,k]
        self.prototypes [k, m, d]
        """
        pred_seg = torch.max(out_seg, 1)[1]  # index [b, h, w]预测的类别
        mask = (gt_seg == pred_seg.view(-1))
        # [n, d], [d, k * m]
        cosine_similarity = torch.mm(_c, self.prototypes.view(-1, self.prototypes.shape[-1]).t())

        proto_logits = cosine_similarity
        proto_target = gt_seg.clone().float()

        # clustering for each class
        protos = self.prototypes.data.clone()
        for k in range(self.num_classes):
            init_q = masks[..., k]  # 第k类的所有prototype
            init_q = init_q[gt_seg == k, ...]  # 选择第k类对应的所有像素点
            if init_q.shape[0] == 0:
                continue

            q, indexs = distributed_sinkhorn(init_q)

            m_k = mask[gt_seg == k]

            c_k = _c[gt_seg == k, ...]

            m_k_tile = repeat(m_k, 'n -> n tile', tile=self.num_prototypes)

            m_q = q * m_k_tile  # n x self.num_prototype

            c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])

            c_q = c_k * c_k_tile  # n x embedding_dim

            f = m_q.transpose(0, 1) @ c_q  # self.num_prototype x embedding_dim

            n = torch.sum(m_q, dim=0)

            if torch.sum(n) > 0:
                f = F.normalize(f, p=2, dim=-1)

                new_value = momentum_update(old_value=protos[k, n != 0, :], new_value=f[n != 0, :],
                                            momentum=self.gamma, debug=False)
                protos[k, n != 0, :] = new_value

            proto_target[gt_seg == k] = indexs.float() + (self.num_prototypes * k)

        self.prototypes = nn.Parameter(l2_normalize(protos),
                                       requires_grad=False)
        return proto_logits, proto_target

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg=None, seg_weight=None):
        out_seg, _c, masks = self.forward(inputs)
        gt_seg = F.interpolate(gt_semantic_seg.float(), size=inputs[0].size()[2:], mode='nearest').view(-1)  # [2 * 512 * 512, ]
        # [2 * 128 * 128, 190], [2 * 128 * 128, ]
        losses = self.losses(out_seg, gt_semantic_seg, _c, gt_seg, masks, seg_weight)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg=None):
        out_seg, _, _ = self.forward(inputs)
        return out_seg

    def forward(self, x):
        """
        Args:
            inputs:list[feat1, feat2, feat3, feat4]
        Returns:

        """
        inputs = x
        _, _, h, w = inputs[0].size()
        feat1 = inputs[0]
        feat2 = F.interpolate(inputs[1], size=(h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(inputs[2], size=(h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(inputs[3], size=(h, w), mode="bilinear", align_corners=True)

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        c = self.cls_head(feats)
        c = self.proj_head(c)
        _c = rearrange(c, 'b c h w -> (b h w) c')
        _c = self.feat_norm(_c)
        _c = l2_normalize(_c)

        self.prototypes.data.copy_(l2_normalize(self.prototypes))
        masks = torch.einsum('nd,kmd->nmk', _c, self.prototypes)  # 将每个像素对应的vector和prototype做点积 [2, 10, 19]
        out_seg = torch.amax(masks, dim=1)  # n,k返回每个类别相似度最高的
        out_seg = self.mask_norm(out_seg)
        out_seg = rearrange(out_seg, "(b h w) k -> b k h w", b=feats.shape[0], h=feats.shape[2])  # [2, 19, 128, 128]

        return out_seg, _c, masks

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label, _c, gt_seg, masks, seg_weight=None):
        contrast_logits, contrast_target = self.prototype_learning(_c, seg_logit, gt_seg, masks)
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        # [2 * 128 * 128, 190], [2 * 128 * 128, ]
        loss = dict()
        seg_label = seg_label.squeeze(1)

        loss['loss_seg'] = self.pixel_loss(seg_logit,
                                           contrast_logits,
                                           contrast_target,
                                           seg_label,
                                           seg_weight)
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        return loss