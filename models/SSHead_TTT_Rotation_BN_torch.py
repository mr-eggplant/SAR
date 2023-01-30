"""
This file support ResNet-50-GroupNorm, from timm pytorch.

Huberyniu, 20220801
"""

from torch import nn
import torch.nn.functional as F
import math
import timm
import copy
import models.Res as Resnet
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

def extractor_from_layer4(net):
	layers = [net.conv1, net.bn1, net.relu, net.maxpool,
				 net.layer1, net.layer2, net.layer3, net.layer4, 
					net.avgpool, ViewFlatten()]
	return nn.Sequential(*layers)

def extractor_from_layer3(net):
	layers = [net.conv1, net.bn1, net.relu, net.maxpool,
				 net.layer1, net.layer2, net.layer3]
	return nn.Sequential(*layers)

def extractor_from_layer2(net):
	layers = [net.conv1, net.bn1, net.relu, net.maxpool,
				 net.layer1, net.layer2]
	return nn.Sequential(*layers)



def build_model(shared_type='layer2'): 

    """
    shared_type: layer2, layer3, layer4
    """

    net = Resnet.__dict__['resnet50'](pretrained=False)
    init = torch.load("/apdcephfs/private_huberyniu/cli_pretrained_models/resnet50-19c8e357.pth")
    net.load_state_dict(init)
	
    width = 1
    expansion = 4
    planes = 512

    if shared_type == 'layer4':
        ext = extractor_from_layer4(net)
        head = nn.Linear(expansion * planes, 4)
    elif shared_type == 'layer3':
        ext = extractor_from_layer3(net)
        head = copy.deepcopy([net.layer4, net.avgpool, 
                                ViewFlatten(), nn.Linear(expansion * planes * width, 4)])
        head = nn.Sequential(*head)
    elif shared_type == 'layer2':
        ext = extractor_from_layer2(net)
        head = copy.deepcopy([net.layer3, net.layer4, net.avgpool, 
                                ViewFlatten(), nn.Linear(expansion * planes * width, 4)])
        head = nn.Sequential(*head)

    ssh = ExtractorHead(ext, head).cuda()

    return net, ext, head, ssh


# net, ext, head, ssh = build_model()

# print(net)