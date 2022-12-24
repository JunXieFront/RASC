import torch
import torch.nn as nn
from mmseg.ops import resize
from ..builder import HEADS, build_loss
from .decode_head import BaseDecodeHead
from .daformer_head import build_layer
from einops import rearrange, repeat
from .proformer_utils import momentum_update, l2_normalize, distributed_sinkhorn
import torch.nn.functional as F
from ..losses import accuracy
from mmcv.runner import force_fp32


@HEADS.register_module()
class ProHead(BaseDecodeHead):

    def __init__(self, entropy_loss, proto_loss, **kwargs):
        super(ProHead, self).__init__(
            input_transform='multiple_select', **kwargs)

        assert not self.align_corners
        self.max_iters = 40000
        self.iters = 0
        self.lamb = None
        decoder_params = kwargs['decoder_params']
        embed_dims = decoder_params['embed_dims']
        if isinstance(embed_dims, int):
            embed_dims = [embed_dims] * len(self.in_index)
        embed_cfg = decoder_params['embed_cfg']
        embed_neck_cfg = decoder_params['embed_neck_cfg']
        if embed_neck_cfg == 'same_as_embed_cfg':
            embed_neck_cfg = embed_cfg
        fusion_cfg = decoder_params['fusion_cfg']
        for cfg in [embed_cfg, embed_neck_cfg, fusion_cfg]:
            if cfg is not None and 'aspp' in cfg['type']:
                cfg['align_corners'] = self.align_corners

        self.embed_layers = {}
        for i, in_channels, embed_dim in zip(self.in_index, self.in_channels,
                                             embed_dims):
            if i == self.in_index[-1]:
                self.embed_layers[str(i)] = build_layer(
                    in_channels, embed_dim, **embed_neck_cfg)
            else:
                self.embed_layers[str(i)] = build_layer(
                    in_channels, embed_dim, **embed_cfg)
        self.embed_layers = nn.ModuleDict(self.embed_layers)

        self.fuse_layer = build_layer(
            sum(embed_dims), self.channels, **fusion_cfg)

        self.entropy_loss = build_loss(entropy_loss)
        self.proto_loss = build_loss(proto_loss)


    def forward_and_fuse(self, inputs):
        x = inputs
        n, _, h, w = x[-1].shape
        # for f in x:
        #     mmcv.print_log(f'{f.shape}', 'mmseg')

        os_size = x[0].size()[2:]
        _c = {}
        for i in self.in_index:
            # mmcv.print_log(f'{i}: {x[i].shape}', 'mmseg')
            _c[i] = self.embed_layers[str(i)](x[i])
            if _c[i].dim() == 3:
                _c[i] = _c[i].permute(0, 2, 1).contiguous()\
                    .reshape(n, -1, x[i].shape[2], x[i].shape[3])
            # mmcv.print_log(f'_c{i}: {_c[i].shape}', 'mmseg')
            if _c[i].size()[2:] != os_size:
                # mmcv.print_log(f'resize {i}', 'mmseg')
                _c[i] = resize(
                    _c[i],
                    size=os_size,
                    mode='bilinear',
                    align_corners=self.align_corners)

        c = self.fuse_layer(torch.cat(list(_c.values()), dim=1))
        # [2, 256, 128, 128]
        return c

    def forward(self, inputs):
        pass

    def forward_train(self, inputs, img_metas, gt_semantic_seg,  train_cfg=None, seg_weight=None, source=True):
        fuse_feature = self.forward_and_fuse(inputs)
        seg_logits = self.cls_seg(fuse_feature)
        if source:
           losses = self.source_loss(seg_logits, gt_semantic_seg, seg_weight)
        else:
            losses = self.target_loss(fuse_feature, seg_logits, gt_semantic_seg, seg_weight)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg=None):
        feat = self.forward_and_fuse(inputs)
        return self.cls_seg(feat)

    def target_loss(self, feat, seg_logit, seg_label, seg_weight=None):
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        seg_label = seg_label.squeeze(1)
        entropy_loss = self.entropy_loss(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)

        mu_s = self.conv_seg.weight.data.clone()
        transfer_loss = self.proto_loss(mu_s, feat)
        loss['loss_proto'] = entropy_loss + transfer_loss
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        return loss


    def source_loss(self, seg_logit, seg_label, seg_weight=None):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        seg_label = seg_label.squeeze(1)
        loss['loss_seg'] = self.entropy_loss(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        return loss
