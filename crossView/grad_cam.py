import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticSegmentationTarget:
    def __init__(self, category):
        self.category = category
        
    def __call__(self, model_output):
        output = F.softmax(model_output, dim=1)
        unknown_p = output[:, 0]
        occ_p = output[:, 1]
        free_p = output[:, 2]
        if self.category == 0:
            mask = (unknown_p >= 0.5)
            return (unknown_p * mask).sum()
        if self.category == 1:
            mask = (occ_p >= 0.5)
            return (occ_p * mask).sum()
        elif self.category == 2:
            mask = (free_p >= 0.5)
            return (free_p * mask).sum()

        raise Exception(f"Invalid category provided {self.category}")


class SegmentationModelOutputWrapper(nn.Module):
    def __init__(self, model): 
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model
        
    def forward(self, x):
        return [self.model(x)]