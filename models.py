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
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden = nn.Linear(1, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        h = self.activation(self.hidden(x))
        return self.output(h)

def get_model(hidden_size, fine_tune_type='full'):
    model = BaseModel(hidden_size)
    
    if fine_tune_type == 'adapter':
        model.adapter = Adapter(hidden_size)
        def forward_with_adapter(self, x):
            h = self.activation(self.hidden(x))
            h = self.adapter(h)
            return self.output(h)
        model.forward = forward_with_adapter.__get__(model)
        
    elif fine_tune_type == 'lora':
        model.lora = LoRALayer(1, hidden_size)
        def forward_with_lora(self, x):
            h = self.activation(self.hidden(x) + self.lora(x))
            return self.output(h)
        model.forward = forward_with_lora.__get__(model)
    
    return model 