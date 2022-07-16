import torch
import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model

# Model list is available : https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/model_provider.py
def select_model(model_name, pretrained):
    model = ptcv_get_model(model_name, pretrained=pretrained)

    return model


class modelEnsemble(nn.Module):   
    def __init__(self, modelA, modelB, modelC, pretrained, out_dim=11):
        super(modelEnsemble, self).__init__()
        self.modelA = select_model(modelA, pretrained)
        self.modelB = select_model(modelB, pretrained)
        self.modelC = select_model(modelC, pretrained)
        self.classifier = nn.Linear(1000 * 3, out_dim)
        
    def forward(self, x):
        x1 = self.modelA(x)
        x2 = self.modelB(x)
        x3 = self.modelC(x)
        x = torch.cat((x1, x2, x3), dim=1)
        out = self.classifier(x)
        
        return out

    def save(self, save_path):
        torch.save({
            'modelA' : self.modelA.state_dict(),
            'modelB' : self.modelB.state_dict(),
            'modelC' : self.modelC.state_dict()
        }, save_path)
    
    def load(self, save_path):
        checkpoint = torch.load(save_path)
        self.modelA.load_state_dict(checkpoint['modelA'])
        self.modelB.load_state_dict(checkpoint['modelB'])
        self.modelC.load_state_dict(checkpoint['modelC'])
