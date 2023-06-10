# -*- coding: utf-8 -*-
import loralib as lora
import torch
import torch.nn as nn


# define a model that contains two lora linear layers
class Model(nn.Module):
    def __init__(self, in_feature, d_dim, n_class):
        super(Model, self).__init__()
        self.layer1 = lora.Linear(in_feature, d_dim, r=16)
        self.layer2 = lora.Linear(d_dim, n_class, r=16)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return self.log_softmax(x)


if __name__ == "__main__":
    # create a model
    # Add a pair of low-rank adaptation matrices with rank r=16
    in_features = 128
    n_class = 2
    d_dim = 64
    model = Model(in_features, d_dim, n_class)

    for name, param in model.named_parameters():
        if "lora" in name:
            print(name, param.shape, param.device, "requires_grad")
            param.requires_grad = True
        else:
            print(name, param.shape, param.device, "not requires_grad")
            param.requires_grad = True

    # fake some input data

    x = torch.randn(16, in_features)
    y = torch.randint(0, n_class, (16,))

    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for i in range(100):
        optimizer.zero_grad()
        output = model(x)
        loss = torch.nn.functional.nll_loss(output, y)
        loss.backward()
        optimizer.step()
        acc = (output.argmax(dim=1) == y).float().mean()
        if i % 10 == 0:
            print(f"i: {i}, loss: {loss.item():.3f}, acc: {acc:.3f}")
