from torch.nn.modules.module import Module
from ..functions.bf_filter import BfFilterFunction

# class Corr_module(Module):
# 	def __init__(self, filter_size, iteration_num):
# 		super(BfFilter, self).__init__()

# 		self.filter_size = int(filter_size)
# 		self.iteration_num = int(iteration_num)

# 	def forward(self, features, weights):
# 		return BfFilterFunction(self.filter_size, 
# 								self.iteration_num)(features, weights)