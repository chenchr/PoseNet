#ifndef _CORR_KERNEL
#define _CORR_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

int CorrForwardLaucher(
	const float* bot0_flat, const float* bot1_flat, float* rbot1_flat, float* rbot2_flat, float* output_flat, 
	const int bnum, const int bchannels, const int bheight, const int bwidth, const int top_channels, const int top_height, const int top_width,
	const int max_displacement, const int kernel_size, const int pad, const int stride_1, const int stride_2, const int corr_type, 
	const int do_abs, const int kernel_radius, const int border_size, const int neighborhood_grid_radius, 
	const int neighborhood_grid_width,
	cudaStream_t stream);

int CorrBackwardLaucher(
    const float* output_flat, float* input0_flat, float* input1_flat, const float* rbot1_flat, const float* rbot2_flat, 
    const int bnum, const int bchannels, const int bheight, const int bwidth, const int top_channels, const int top_height, const int top_width,
	const int max_displacement, const int kernel_size, const int pad, const int stride_1, const int stride_2, const int corr_type, 
	const int do_abs, const int kernel_radius, const int border_size, const int neighborhood_grid_radius, 
	const int neighborhood_grid_width,
	cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif

