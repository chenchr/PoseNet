#include <THC/THC.h>
#include <math.h>
#include "cuda/corr_kernel.h"
#include <stdio.h>

extern THCState *state;

int corr_forward_cuda(THCudaTensor *bot0, THCudaTensor *bot1, THCudaTensor *rbot1, THCudaTensor *rbot2, THCudaTensor *output,
                      int max_displacement, int kernel_size, int pad, int stride_1, int stride_2, int corr_type, 
                      int do_abs, int kernel_radius, int border_size, int neighborhood_grid_radius, int neighborhood_grid_width){
	// Grab the input tensor
    float *bot0_flat = THCudaTensor_data(state, bot0);
    float *bot1_flat = THCudaTensor_data(state, bot1);
    float *rbot1_flat = THCudaTensor_data(state, rbot1);
    float *rbot2_flat = THCudaTensor_data(state, rbot2);
    float *output_flat = THCudaTensor_data(state, output);

    const int bnum = THCudaTensor_size(state, bot0, 0);
    const int bchannels = THCudaTensor_size(state, bot0, 1);
    const int bheight = THCudaTensor_size(state, bot0, 2);
    const int bwidth = THCudaTensor_size(state, bot0, 3);
    
    const int top_channels = THCudaTensor_size(state, output, 1);
    const int top_height = THCudaTensor_size(state, output, 2);
    const int top_width = THCudaTensor_size(state, output, 3);

    cudaStream_t stream = THCState_getCurrentStream(state);

    CorrForwardLaucher(
        bot0_flat, bot1_flat, rbot1_flat, rbot2_flat, output_flat,
        bnum, bchannels, bheight, bwidth, top_channels, top_height, top_width,
        max_displacement, kernel_size, pad, stride_1, stride_2, corr_type, do_abs, kernel_radius, border_size,
        neighborhood_grid_radius, neighborhood_grid_width, stream);
    return 1;
}

int corr_backward_cuda(THCudaTensor *grad_output, THCudaTensor *grad_input0, THCudaTensor *grad_input1, THCudaTensor *rbot1, THCudaTensor *rbot2,
                      int max_displacement, int kernel_size, int pad, int stride_1, int stride_2, int corr_type, int do_abs, 
                      int kernel_radius, int border_size, int neighborhood_grid_radius, int neighborhood_grid_width){
	// Grab the input tensor
    float *output_flat = THCudaTensor_data(state, grad_output);
    float *input0_flat = THCudaTensor_data(state, grad_input0);
    float *input1_flat = THCudaTensor_data(state, grad_input1);
    float *rbot1_flat = THCudaTensor_data(state, rbot1);
    float *rbot2_flat = THCudaTensor_data(state, rbot2);

    const int bnum = THCudaTensor_size(state, grad_input0, 0);
    const int bchannels = THCudaTensor_size(state, grad_input0, 1);
    const int bheight = THCudaTensor_size(state, grad_input0, 2);
    const int bwidth = THCudaTensor_size(state, grad_input0, 3);
    
    const int top_channels = THCudaTensor_size(state, grad_output, 1);
    const int top_height = THCudaTensor_size(state, grad_output, 2);
    const int top_width = THCudaTensor_size(state, grad_output, 3);

    cudaStream_t stream = THCState_getCurrentStream(state);

    CorrBackwardLaucher(
        output_flat, input0_flat, input1_flat, rbot1_flat, rbot2_flat, 
        bnum, bchannels, bheight, bwidth, top_channels, top_height, top_width,
        max_displacement, kernel_size, pad, stride_1, stride_2, corr_type, do_abs, kernel_radius, border_size,
        neighborhood_grid_radius, neighborhood_grid_width, stream);
    return 1;
}