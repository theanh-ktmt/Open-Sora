import torch
import torch.nn as nn
from torchviz import make_dot


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 5)

    def forward(self, x):
        x = self.fc1(x)  # Shape: [batch_size, 10]
        x = torch.matmul(x.unsqueeze(2), x.unsqueeze(1))  # Shape: [batch_size, 10, 10]
        x = torch.flatten(x, start_dim=1)  # Flatten to shape: [batch_size, 100]
        x = self.fc2(x[:, :10])  # Shape: [batch_size, 5], only using first 10 features for simplicity
        return x


model = MyModel()
example_input = torch.randn(1, 10)  # Example input: [batch_size, input_features]

# Forward pass to create the graph
y = model(example_input)

# Visualize the graph
dot = make_dot(y, params=dict(model.named_parameters()))
dot.format = "png"
dot.render("model_graph")
