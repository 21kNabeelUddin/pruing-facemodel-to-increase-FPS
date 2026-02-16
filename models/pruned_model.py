import torch
import torch.nn as nn
from .base_model import get_base_model
import torch.nn.functional as F

class PrunedInceptionResnetV1(nn.Module):
    def __init__(self, original_model, skip_config={}):
        """
        skip_config: dict mapping 'repeat_1', 'repeat_2', 'repeat_3' to list of block indices.
        """
        super(PrunedInceptionResnetV1, self).__init__()
        self.mk = original_model # mk = model kernel / backbone
        self.skip_config = skip_config

    def forward(self, x):
        x = self.mk.conv2d_1a(x)
        x = self.mk.conv2d_2a(x)
        x = self.mk.conv2d_2b(x)
        x = self.mk.maxpool_3a(x)
        x = self.mk.conv2d_3b(x)
        x = self.mk.conv2d_4a(x)
        x = self.mk.conv2d_4b(x)
        
        # Repeat 1
        x = self._forward_repeat(x, 'repeat_1', self.mk.repeat_1)
        
        x = self.mk.mixed_6a(x)
        
        # Repeat 2
        x = self._forward_repeat(x, 'repeat_2', self.mk.repeat_2)
        
        x = self.mk.mixed_7a(x)
        
        # Repeat 3
        x = self._forward_repeat(x, 'repeat_3', self.mk.repeat_3)
        
        x = self.mk.block8(x)
        x = self.mk.avgpool_1a(x)
        x = self.mk.dropout(x)
        
        # KEY FIX: Flatten before linear layer
        x = x.view(x.shape[0], -1)
        
        x = self.mk.last_linear(x)
        x = self.mk.last_bn(x)
        
        
        # x = self.mk.logits(x) # Facenet-pytorch specific logic if classify=False usually returns embeddings
        
        
        return x

    def _forward_repeat(self, x, layer_name, layer_module):
        skip_indices = self.skip_config.get(layer_name, [])
        for i, block in enumerate(layer_module):
            if i in skip_indices:
                continue
            x = block(x)
        return x

def get_pruned_model(skip_config={}):
    original = get_base_model()
    return PrunedInceptionResnetV1(original, skip_config)
