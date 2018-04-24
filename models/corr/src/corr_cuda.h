int corr_forward_cuda(THCudaTensor *bot0, THCudaTensor *bot1, THCudaTensor *rbot1, THCudaTensor *rbot2, THCudaTensor *output,
					  int max_displacement, int kernel_size, int pad, int stride_1, int stride_2, int corr_type, int do_abs, 
					  int kernel_radius, int border_size, int neighborhood_grid_radius, int neighborhood_grid_width);

int corr_backward_cuda(THCudaTensor *grad_output, THCudaTensor *grad_input0, THCudaTensor *grad_input1, THCudaTensor *rbot1, THCudaTensor *rbot2,
					  int max_displacement, int kernel_size, int pad, int stride_1, int stride_2, int corr_type, int do_abs, 
					  int kernel_radius, int border_size, int neighborhood_grid_radius, int neighborhood_grid_width);