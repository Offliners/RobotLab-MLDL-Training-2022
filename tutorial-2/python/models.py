from pytorchcv.model_provider import get_model as ptcv_get_model

# Model list is available : https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/model_provider.py
def select_model(model_name, pretrained):
    model = ptcv_get_model(model_name, pretrained=pretrained)

    return model