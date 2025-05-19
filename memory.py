# -*- coding: utf-8 -*-
"""
MemoryDNN: PyTorch-based DNN for memory.
This module implements a memory network using a simple feed-forward neural network.
Based on the original LyDROO implementation.
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MemoryDNN(nn.Module):
    def __init__(self, net, learning_rate=0.01, training_interval=10, batch_size=100, memory_size=1000):
        """
        net: list specifying the network architecture, e.g., [input_dim, hidden1, hidden2, output_dim]
        """
        super(MemoryDNN, self).__init__()
        self.net = net
        self.training_interval = training_interval
        self.batch_size = batch_size
        self.memory_size = memory_size
        
        # Build the feed-forward network
        layers_list = []
        input_dim = net[0]
        for hidden_dim in net[1:-1]:
            layers_list.append(nn.Linear(input_dim, hidden_dim))
            layers_list.append(nn.ReLU())
            input_dim = hidden_dim
        layers_list.append(nn.Linear(input_dim, net[-1]))
        layers_list.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers_list)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        
        # Initialize memory to store (h, m) pairs
        self.memory = np.zeros((memory_size, net[0] + net[-1]))
        self.memory_counter = 0
        self.cost_his = []
    
    def forward(self, x):
        return self.model(x)
    
    def remember(self, h, m):
        # h: numpy array of shape (input_dim,)
        # m: numpy array of shape (output_dim,)
        idx = self.memory_counter % self.memory_size
        self.memory[idx, :] = np.hstack((h, m))
        self.memory_counter += 1
    
    def encode(self, h, m):
        self.remember(h, m)
        if self.memory_counter % self.training_interval == 0:
            self.learn()
    
    def learn(self):
        if self.memory_counter < self.batch_size:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size, replace=True)
        elif self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size, replace=False)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size, replace=False)
        batch_memory = self.memory[sample_index, :]
        input_dim = self.net[0]
        h_train = torch.tensor(batch_memory[:, :input_dim], dtype=torch.float32)
        m_train = torch.tensor(batch_memory[:, input_dim:], dtype=torch.float32)
        self.optimizer.zero_grad()
        outputs = self.model(h_train)
        loss = self.criterion(outputs, m_train)
        loss.backward()
        self.optimizer.step()
        self.cost_his.append(loss.item())
    
    def decode(self, h, k=1, decoder_mode=None):
        # h: numpy array of shape (input_dim,)
        h_tensor = torch.tensor(h, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            m_pred = self.model(h_tensor).squeeze(0).numpy()
        # Generate a candidate binary action from m_pred (threshold at 0.5)
        m_candidate = (m_pred > 0.5).astype(int)
        return (m_pred, [m_candidate])
    
    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(np.arange(len(self.cost_his)) * self.training_interval, self.cost_his)
        plt.ylabel('Training Loss')
        plt.xlabel('Time Frames')
        plt.title('MemoryDNN Training Cost')
        plt.show()
