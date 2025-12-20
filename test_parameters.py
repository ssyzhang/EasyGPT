
from model import GPT,GPTConfig
import torch.nn as nn
config = GPTConfig()

mymodel=GPT(config)


num_config=mymodel.get_num_params()

print("Number of parameters: %.2fM" % (num_config/1e6,))



# class TestModule(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear1 = nn.Linear(10, 10, bias=False)
#         self.linear2 = nn.Linear(10, 10, bias=False)
#         self.linear1.weight=self.linear2.weight  # share weights
#     def get_num_params(self):
#         return sum(p.numel() for p in self.parameters())
    
#     def forward(self, x):
#         x = self.linear1(x)
#         x = self.linear2(x)
#         return x
#     def show_params(self):
#         for param in self.parameters():
#             print(f"param: {param}")
    
# test_model = TestModule()
# num_test_params = test_model.get_num_params()
# print(f"Number of parameters in test model: {num_test_params}")
# test_model.show_params()