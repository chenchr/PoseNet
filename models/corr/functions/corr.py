from __future__ import print_function
import torch
from torch.autograd import Function
from .._ext import corr
import math

class CorrFunction(Function):
	def __init__(self, max_displacement, kernel_size, pad, stride_1, stride_2, corr_type=0, do_abs= 0):
		super(CorrFunction, self).__init__()

		self.max_displacement = int(max_displacement)
		self.kernel_size = int(kernel_size)
		if kernel_size%2 == 0:
			print("only support odd kernel size!")
			return
		self.pad = int(pad)
		self.stride_1 = int(stride_1)
		self.stride_2 = int(stride_2)
		self.corr_type = int(corr_type)
		self.do_abs = int(do_abs)

	def forward(self, bot0, bot1):
		# print("features: ", features.sum())
		# print(features)
		batch_size, num_channels, data_height, data_width = bot0.shape
		self.botshape = bot0.shape

		paddedbotheight = long(data_height+2*self.pad)
		paddedbotwidth = long(data_width+2*self.pad)

		self.kernel_radius = (self.kernel_size-1)//2
		self.border_size = self.max_displacement+self.kernel_radius

		top_width = long(math.ceil(float(paddedbotwidth-self.border_size*2)/float(self.stride_1)))
		top_height = long(math.ceil(float(paddedbotheight-self.border_size*2)/float(self.stride_1)))

		self.neighborhood_grid_radius = int(self.max_displacement//self.stride_2)
		self.neighborhood_grid_width = int(self.neighborhood_grid_radius*2+1)

		top_channels = long(self.neighborhood_grid_width*self.neighborhood_grid_width)

		output = torch.zeros(batch_size, top_channels, top_height, top_width).cuda().contiguous()
		rbot1 = torch.zeros(batch_size, paddedbotheight, paddedbotwidth,num_channels).cuda().contiguous()
		rbot2 = torch.zeros(batch_size, paddedbotheight, paddedbotwidth,num_channels).cuda().contiguous()

		self.rbot1 = rbot1
		self.rbot2 = rbot2

		if not (bot0.is_cuda and bot1.is_cuda):
			raise NotImplementedError
		bot0, bot1 = bot0.contiguous(), bot1.contiguous()
		corr.corr_forward_cuda(bot0, bot1, rbot1, rbot2, output, self.max_displacement, self.kernel_size, self.pad, 
			self.stride_1, self.stride_2, self.corr_type, self.do_abs, self.kernel_radius, self.border_size,
			self.neighborhood_grid_radius, self.neighborhood_grid_width)
		# print("rbot1: ", rbot1)
		# print("rbot2", rbot2)
		return output

	def backward(self, grad_output):
		if not grad_output.is_cuda:
			raise NotImplementedError

		# print('grad output: \n')
		# print(grad_output)

		batch_size, num_channels, data_height, data_width = self.botshape


		grad_input0 = torch.zeros(batch_size, num_channels, 
								 data_height, data_width).cuda().contiguous()
		grad_input1 = torch.zeros(batch_size, num_channels, 
								 data_height, data_width).cuda().contiguous()

		grad_output = grad_output.contiguous()
		corr.corr_backward_cuda(grad_output, grad_input0, grad_input1, self.rbot1, self.rbot2, self.max_displacement, self.kernel_size, self.pad, 
			self.stride_1, self.stride_2, self.corr_type, self.do_abs, self.kernel_radius, self.border_size,
			self.neighborhood_grid_radius, self.neighborhood_grid_width)
		# print("grad0: \n")
		# print(grad_input0)
		# print("grad1: \n")
		# print(grad_input1)
		return grad_input0, grad_input1