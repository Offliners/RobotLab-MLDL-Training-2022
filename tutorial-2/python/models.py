import torch
import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model

# Model list is available : https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/model_provider.py
def select_model(model_name, pretrained=False):
    model = ptcv_get_model(model_name, pretrained=pretrained)

    return model


class modelEnsemble(nn.Module):   
    def __init__(self, modelA, modelB, modelC, out_dim=11):
        super(modelEnsemble, self).__init__()
        self.modelA = select_model(modelA)
        self.modelB = select_model(modelB)
        self.modelC = select_model(modelC)
        self.classifier = nn.Linear(1000 * 3, out_dim)
        
    def forward(self, x):
        x1 = self.modelA(x)
        x2 = self.modelB(x)
        x3 = self.modelC(x)
        x = torch.cat((x1, x2, x3), dim=1)
        out = self.classifier(x)
        
        return out