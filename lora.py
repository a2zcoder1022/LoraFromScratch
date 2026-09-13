import torch.nn as nn
import torch
import torch.nn.functional as F
from LoraModel import LinearLayer

class Lora(nn.Module):
    def __init__(self, model, r: int, alpha: int, dropout: float, target_weights: list):
        super().__init__()
        self.target_weight_modules = []
        self.model = model
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        self.target_weights = target_weights
        self.freeze_model_weights()
        self.review_weight_tree()


    def review_weight_tree(self):
         if len(self.target_weights) == 0 or len(self.target_weights) > 1:
             raise ValueError("Target weights must be a list of one name")
         
         for name, model in self.model.named_modules():
             abbrev_name = name.split(".")[-1]
             if isinstance(model, nn.Linear) and abbrev_name in self.target_weights:
                 self.target_weight_modules.append((name, model))
        


    def validate_rank(self, original_weights):
        if self.r <= 0 or self.r > min(original_weights.in_features, original_weights.out_features):
            raise ValueError("Rank is not valid")


    def freeze_model_weights(self):
        for name, param in self.model.named_parameters():
                param.requires_grad = False


    def update_module(self):
        for name, module in self.target_weight_modules:
            original_weights = self.model.get_submodule(name)
            self.validate_rank(original_weights)
            lora_module = LinearLayer(original_weights, self.r, self.alpha, self.dropout)
            self.model.set_submodule(name, lora_module)
        return self.model