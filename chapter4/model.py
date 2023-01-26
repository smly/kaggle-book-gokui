import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from torch.nn.parameter import Parameter


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(
        x.clamp(min=eps).pow(p),
        (x.size(-2), x.size(-1))
    ).pow(1./p)


class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(
            self.p.data.tolist()[0]
        ) + ', ' + 'eps=' + str(self.eps) + ')'


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        input = input.float()

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class AngularModel(nn.Module):

    def __init__(self, n_classes=7770, model_name="resnet34", pooling="GeM",
                 margin=0.3, scale=30, fc_dim=512,
                 pretrained=None, loss_kwargs=None):
        super(AngularModel, self).__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
        )
        final_in_features = self.backbone.fc.in_features

        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        loss_kwargs = loss_kwargs or {
            "s": scale,
            "m": margin,
            "easy_margin": False,
            "ls_eps": 0.0,
        }

        self.pooling = GeM()
        self.dropout = nn.Dropout(p=0.0)
        self.fc = nn.Linear(final_in_features, fc_dim)
        self.bn = nn.BatchNorm1d(fc_dim)
        self._init_params()

        self.final = ArcMarginProduct(fc_dim,
                                      n_classes,
                                      **loss_kwargs)

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x, label):
        feature = self.extract_features(x)
        logits = self.final(feature, label)
        return logits

    def extract_features(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.pooling(x).view(batch_size, -1)

        # fc
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn(x)

        return x
