
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, datasets, transforms
import torch
import numpy as np


class FDNet(nn.Module):
    def __init__(self, out_features=2):
        super(FDNet, self).__init__()
        mnet = models.mobilenet_v2(pretrained=True)
        for name, param in mnet.named_parameters():
            if("bn" not in name):
                param.requires_grad_(False)

        # Parameters of newly constructed modules have requires_grad=True by default
        in_features = mnet.classifier[1].in_features
        mnet.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False, ),
            nn.Linear(in_features, 500),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(500, out_features))

        self.mnet = mnet

    def forward(self, images):
        features = self.mnet(images)
        if not self.training:
            features = torch.sigmoid(features)
        return features


def load_fd_net_model(state_dict_path='fdnet.pt'):
    fd_net_model = FDNet()
    fd_net_model.load_state_dict(torch.load(state_dict_path))
    # del fd_net_model.mnet.classifier[0]
    # del fd_net_model.mnet.classifier[2]
    # print(fd_net_model.mnet.classifier)
    return fd_net_model


def _preprocess(X):
    preprocessed = np.array(X, 'f4')
    preprocessed = preprocessed / 255
    preprocessed = preprocessed - np.array([0.485, 0.456, 0.406], 'f4')
    preprocessed = preprocessed / np.array([0.229, 0.224, 0.225], 'f4')
    preprocessed = preprocessed.transpose(0, 3, 1, 2)
    return preprocessed


def inference(model, X, split_batch_size=None, preprocess=False):
    if preprocess:
        X = _preprocess(X)

    X = torch.from_numpy(X).to(next(model.parameters()).device)
    if split_batch_size is not None:
        X = [X[i * split_batch_size: (i + 1) * split_batch_size]
             for i in range(X.shape[0] // split_batch_size + 1)]
        X = [x for x in X if x.size]
        Y = []
        for x in X:
            y = model(x).cpu().detach().numpy()
            Y.append(y)
        Y = np.concatenate(Y, axis=0)
    else:
        Y = model(X).cpu().detach().numpy()
    return Y
