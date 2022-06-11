import torch
import torchvision
from torch import nn

from models.model_MLP import MLP


class Dense_feature(nn.Module):
    def __init__(self, drop_out = 0.5, *args, ** kwargs):
        self.args = kwargs['args']
        module = self.args.arch
        mode = self.args.mode
        num_classes = self.args.num_classes

        super(Dense_feature, self).__init__()
        if module=='densenet121':
            self.pretrain = torchvision.models.densenet121(pretrained=True).cuda()
        elif module=='densenet161':
            self.pretrain = torchvision.models.densenet161(pretrained=True).cuda()
        elif module=='densenet169':
            self.pretrain = torchvision.models.densenet169(pretrained=True).cuda()
        elif module=='densenet201':
            self.pretrain = torchvision.models.densenet201(pretrained=True).cuda()
        else:
            raise ValueError

        if mode == 'audio':
            self.pretrain.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        dim_mlp = self.pretrain.classifier.in_features
        dim = dim_mlp
        self.pretrain.classifier = nn.Identity()

        self.out = MLP([dim, dim/2, dim/4, num_classes], drop_out=drop_out)

    def forward(self, x):
        # self.pretrain.eval()
        return self.out(self.pretrain(x))


