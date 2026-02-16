import torch
import torch.nn as nn
from .base_model import get_base_model
import copy

class PrunedIResNet(nn.Module):
    def __init__(self, original_model, skip_config={}):
        """
        skip_config: dict mapping layer names to list of block indices to skip.
        e.g., {'layer2': [1, 3], 'layer3': [0, 2, 4]}
        """
        super(PrunedIResNet, self).__init__()
        self.features = original_model
        
        # We need to monkey-patch or wrap the layers we want to prune
        self.skip_config = skip_config
        
        # Create a mapping of layer names to actual sequential modules
        self.layers = {
            'layer1': self.features.layer1,
            'layer2': self.features.layer2,
            'layer3': self.features.layer3,
            'layer4': self.features.layer4
        }

    def forward(self, x):
        x = self.features.conv1(x)
        x = self.features.bn1(x)
        x = self.features.prelu(x)
        
        # Process layer1
        x = self._forward_layer(x, 'layer1')
        
        # Process layer2
        x = self._forward_layer(x, 'layer2')
        
        # Process layer3
        x = self._forward_layer(x, 'layer3')
        
        # Process layer4
        x = self._forward_layer(x, 'layer4')
        
        x = self.features.bn2(x)
        x = torch.flatten(x, 1)
        x = self.features.dropout(x)
        x = self.features.fc(x)
        x = self.features.features(x)

        return x

    def _forward_layer(self, x, layer_name):
        if layer_name not in self.layers:
            return x
            
        layer = self.layers[layer_name]
        skip_indices = self.skip_config.get(layer_name, [])
        
        for i, block in enumerate(layer):
            if i in skip_indices:
                continue
            x = block(x)
            
        return x

def get_pruned_model(base_model_name='r50', skip_config={}):
    original = get_base_model(base_model_name)
    return PrunedIResNet(original, skip_config)
