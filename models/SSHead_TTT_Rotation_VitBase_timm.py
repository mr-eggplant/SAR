"""
This file support ResNet-50-GroupNorm, from timm pytorch.

Huberyniu, 20220801
"""

from torch import nn
import torch.nn.functional as F
import math
import timm
import copy
import torch


class ViewFlatten(nn.Module):
	def __init__(self):
		super(ViewFlatten, self).__init__()

	def forward(self, x):
		return x.view(x.size(0), -1)


class ExtractorHead(nn.Module):
	def __init__(self, ext, head):
		super(ExtractorHead, self).__init__()
		self.ext = ext
		self.head = head

	def forward(self, x):
		return self.head(self.ext(x))


class Pos_Embded(nn.Module):
    def __init__(self, no_embed_class, cls_token, pos_embed, pos_drop):
        super(Pos_Embded, self).__init__()
        self.no_embed_class = no_embed_class
        self.cls_token = cls_token
        self.pos_embed = pos_embed
        self.pos_drop = pos_drop

    def forward(self, x):
        # x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        # x = x + self.pos_embed
        # return self.pos_drop(x)
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed
        return self.pos_drop(x)


class ObtainToken(nn.Module):
	def __init__(self):
		super(ObtainToken, self).__init__()

	def forward(self, x):
		return x[:,0,:].squeeze(dim=1)


def extractor_from_which_block(net, block_num=6):
    layers = [net.patch_embed, Pos_Embded(net.no_embed_class, net.cls_token, net.pos_embed, net.pos_drop), net.blocks[:block_num]]
    return nn.Sequential(*layers)


def build_model(block_num=6): 

    """
    shared_type: block1 2 3 4 5 6 ... 11
    """

    net = timm.create_model('vit_base_patch16_224', pretrained=True)

    # print(net)

    ext = extractor_from_which_block(net, block_num=block_num)
    # head = copy.deepcopy([net.blocks[block_num:], net.norm, ObtainToken(), net.fc_norm, net.head])
    head = copy.deepcopy([net.blocks[block_num:], net.norm, ObtainToken(), net.fc_norm, nn.Linear(768, 4)])
    head = nn.Sequential(*head)

    ssh = ExtractorHead(ext, head)
    return net, ext, head, ssh

# net, ext, head, ssh = build_model()

# print(head)

# print(net)