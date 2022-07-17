import torch
import torch.nn as nn

# Model list is available : https://pytorch.org/vision/0.11/models.html
def select_model(model_name, pretrained=False, num_classes=11):
    model = torch.hub.load('pytorch/vision:v0.11.0', model_name, pretrained=pretrained, num_classes=num_classes)

    return model


class modelEnsemble(nn.Module):   
    def __init__(self, save_names, save_paths):
        super(modelEnsemble, self).__init__()
        self.modelA = select_model(save_names[0])
        self.modelA.load_state_dict(torch.load(save_paths[0]))
        self.modelB = select_model(save_names[1])
        self.modelB.load_state_dict(torch.load(save_paths[1]))
        self.modelC = select_model(save_names[2])
        self.modelC.load_state_dict(torch.load(save_paths[2]))
        
    def forward(self, x):
        x1 = self.modelA(x)
        x2 = self.modelB(x)
        x3 = self.modelC(x)
        
        return (x1 + x2 + x3)