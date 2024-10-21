import torch

inputs = [torch.randn((1, 3, 224, 224)).cuda()]  # your inputs go here

# You can run this in a new python session!
model = torch.export.load("trt.ep").module()
# model = torch_tensorrt.load("trt.ep").module() # this also works
model(*inputs)
