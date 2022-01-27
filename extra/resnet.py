from detectron2.layers import ShapeSpec, FrozenBatchNorm2d
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.resnet import build_resnet_backbone
from detectron2.modeling.backbone.fpn import LastLevelMaxPool, LastLevelP6P7


import torch
import torchvision.models as models
from torchvision.models._utils import IntermediateLayerGetter

from fpn import FPN
from dyhead import DyHead
from byol import BYOL

def get_byol_backbone(cfg):
    backbone = models.wide_resnet50_2(pretrained=True)
    model = BYOL(backbone)
    model_dict = model.state_dict()
    pretrained_dict = torch.load(cfg.MODEL.MODEL_CHECKPOINT, map_location='cpu')
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)
    return model

@BACKBONE_REGISTRY.register()
def build_retinanet_resnet_fpn_dyhead_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    in_channels_p6p7 = out_channels
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7(in_channels_p6p7, out_channels, 'p5'),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    dyhead = DyHead(cfg, backbone)
    return dyhead


@BACKBONE_REGISTRY.register()
def build_resnet_fpn_dyhead_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    # bottom_up = build_resnet_backbone(cfg, input_shape)
    
    return_layers = {"4":"res2", "5": "res3",
                     "6": "res4", "7": "res5"}

    resnet =  get_byol_backbone(cfg)
    frozen_range = [resnet.backbone[0], resnet.backbone[4]]
    for module in frozen_range:
         for param in module.parameters():
             param.requires_grad = False
    resnet = FrozenBatchNorm2d.convert_frozen_batchnorm(resnet)
    bottom_up = IntermediateLayerGetter(resnet.backbone, return_layers)

    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        return_layers=return_layers,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    dyhead = DyHead(cfg, backbone)
    return dyhead