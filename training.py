import torch
import torch.optim as optim
from typing import Tuple, Dict, Optional, List
from torch.utils.data import TensorDataset, DataLoader

class Trainer:
    def __init__(self, model, fine_tune_method='full', learning_rate=0.001):
        self.model = model
        self.fine_tune_method = fine_tune_method
        self.learning_rate = learning_rate
        self.optimizer = self.configure_optimizer()
        self.criterion = torch.nn.MSELoss()
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def configure_optimizer(self):
        if self.fine_tune_method == 'full':
            return optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.fine_tune_method == 'adapter':
            return optim.Adam(self.model.adapter.parameters(), lr=self.learning_rate)
        elif self.fine_tune_method == 'lora':
            return optim.Adam(self.model.lora.parameters(), lr=self.learning_rate)
    
    def train(self, data, batch_size=64, epochs=100):
        x_train = data['train']['x']
        y_train = data['train']['y']
        x_val = data['val']['x']
        y_val = data['val']['y']
        
        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                self.optimizer.zero_grad()
                output = self.model(batch_x)
                loss = self.criterion(output, batch_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_output = self.model(x_val)
                val_loss = self.criterion(val_output, y_val).item()
            
            history['train_loss'].append(train_loss / len(train_loader))
            history['val_loss'].append(val_loss)
        
        return history 