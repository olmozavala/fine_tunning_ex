import torch
import torch.nn as nn

class Adapter(nn.Module):
    def __init__(self, size, reduction_factor=4):
        super().__init__()
        self.down = nn.Linear(size, size // reduction_factor)
        self.up = nn.Linear(size // reduction_factor, size)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.up(self.activation(self.down(x))) + x

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.scaling = 0.1
        
    def forward(self, x):
        return self.lora_B(self.lora_A(x)) * self.scaling

class BaseModel(nn.Module):
    def __init__(self, hidden_size, n_layers=1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(1, hidden_size))
        self.bn_layers.append(nn.BatchNorm1d(hidden_size))
        
        # Hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.bn_layers.append(nn.BatchNorm1d(hidden_size))
            
        # Output layer
        self.output = nn.Linear(hidden_size, 1)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        for layer, bn in zip(self.layers, self.bn_layers):
            x = layer(x)
            x = bn(x)
            x = self.activation(x)
        return self.output(x)

def get_model(hidden_size, fine_tune_type='none', n_layers=1):
    model = BaseModel(hidden_size, n_layers)
    
    # First ensure all parameters are unfrozen
    for param in model.parameters():
        param.requires_grad = True
    
    if fine_tune_type == 'none':
        # For 'none', we don't freeze parameters during initial training
        pass  # Remove the freezing code here
    elif fine_tune_type == 'adapter':
        model.adapter = Adapter(hidden_size)
        def forward_with_adapter(self, x):
            for layer, bn in zip(self.layers, self.bn_layers):
                x = layer(x)
                x = bn(x)
                x = self.activation(x)
            x = self.adapter(x)
            return self.output(x)
        model.forward = forward_with_adapter.__get__(model)
        
    elif fine_tune_type == 'lora':
        model.lora = LoRALayer(1, hidden_size)
        def forward_with_lora(self, x):
            x = self.activation(self.bn_layers[0](self.layers[0](x) + self.lora(x)))
            for layer, bn in list(zip(self.layers, self.bn_layers))[1:]:
                x = layer(x)
                x = bn(x)
                x = self.activation(x)
            return self.output(x)
        model.forward = forward_with_lora.__get__(model)
    
    elif fine_tune_type == 'freeze':
        # Freeze first 10 layers and their batch norms
        for i in range(min(10, len(model.layers))):
            for param in model.layers[i].parameters():
                param.requires_grad = False
            for param in model.bn_layers[i].parameters():
                param.requires_grad = False
    
    return model 