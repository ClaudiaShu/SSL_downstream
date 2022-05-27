import torch
import torchvision
from torch import nn

from models.model_MLP import MLP


class RES_feature(nn.Module):
    def __init__(self, module='resnet50', dim=2048, num_classes=8, drop_out=0.5):
        super(RES_feature, self).__init__()
        if module=='resnet18':
            self.pretrain = torchvision.models.resnet18(pretrained=True).cuda()
        elif module=='resnet50':
            self.pretrain = torchvision.models.resnet50(pretrained=True).cuda()
        elif module=='resnet101':
            self.pretrain = torchvision.models.resnet101(pretrained=True).cuda()
        elif module=='resnet152':
            self.pretrain = torchvision.models.resnet152(pretrained=True).cuda()
        else:
            self.pretrain = torchvision.models.resnet34(pretrained=True).cuda()


        dim_mlp = self.pretrain.fc.in_features
        dim = dim_mlp
        self.pretrain.fc = nn.Identity()

        # dim = self.pretrain.fc.out_features

        # add mlp projection head
        # self.pretrain.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.pretrain.fc)

        self.out = MLP([dim, dim/2, dim/4, num_classes], drop_out=drop_out)

    def forward(self, x):
        # self.pretrain.eval()
        return self.out(self.pretrain(x))


