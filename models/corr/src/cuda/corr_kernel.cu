#include <THC.h>
#include <THCGeneral.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "corr_kernel.h"
#include <stdio.h>
#include <math.h> 
#include <time.h>

#define ROUND_OFF 50000

#define WARPS_PER_BLOCK 1
#define THREADS_PER_WARP 32

__global__ 
void blob_rearrange_kernel2(const float* in, float* out, int num, int channels, 
							int width, int height, int widthheight, int padding, int pwidthheight)
{
    int xy = blockIdx.x*blockDim.x + threadIdx.x;
    if(xy>=widthheight)
        return;

    int ch = blockIdx.y;
    int n  = blockIdx.z;

    float value=in[(n*channels+ch)*widthheight+xy];

    __syncthreads();

    int xpad  = (xy % width + padding);
    int ypad  = (xy / width + padding);
    int xypad = ypad * (width+2*padding) + xpad;

    out[(n*pwidthheight+xypad)*channels + ch] = value;
}

// == Correlation Kernel
__global__ 
void CorrelateData(const int nthreads, int num, int topwidth, int topheight, int topchannels, int topcount,
  int max_displacement, int neighborhood_grid_radius, int neighborhood_grid_width, int kernel_radius, int kernel_size, int stride1, int stride2,
  int bottomwidth, int bottomheight, int bottomchannels,
  const float *bottom0, const float *bottom1, float *top) 
{
  extern __shared__ char patch_data_char[];
  
  float *patch_data = (float *)patch_data_char;
  
    // First (upper left) position of kernel upper-left corner in current center position of neighborhood in image 1
  int x1 = blockIdx.x*stride1 + max_displacement;
  int y1 = blockIdx.y*stride1 + max_displacement;
  int item = blockIdx.z;
  int ch_off = threadIdx.x;
  
  // Load 3D patch into shared shared memory
  for(int j = 0; j < kernel_size; j++) { // HEIGHT
    for(int i = 0; i < kernel_size; i++) { // WIDTH
      int ji_off = ((j * kernel_size) + i) * bottomchannels;
      for(int ch = ch_off; ch < bottomchannels; ch += (WARPS_PER_BLOCK*THREADS_PER_WARP)) { // CHANNELS
          int idx1 = ((item * bottomheight + y1+j) * bottomwidth + x1+i) * bottomchannels + ch;
          int idxPatchData = ji_off + ch;
          patch_data[idxPatchData] = bottom0[idx1];
      }
    }
  }
  
  __syncthreads();
  
  __shared__ float sum[WARPS_PER_BLOCK*THREADS_PER_WARP];
  
  // Compute correlation
  for(int top_channel = 0; top_channel < topchannels; top_channel++) {
    sum[ch_off] = 0;
  
    int s2o = (top_channel % neighborhood_grid_width - neighborhood_grid_radius) * stride2;
    int s2p = (top_channel / neighborhood_grid_width - neighborhood_grid_radius) * stride2;
    
    for(int j = 0; j < kernel_size; j++) { // HEIGHT
      for(int i = 0; i < kernel_size; i++) { // WIDTH
        int ji_off = ((j * kernel_size) + i) * bottomchannels;
        for(int ch = ch_off; ch < bottomchannels; ch += (WARPS_PER_BLOCK*THREADS_PER_WARP)) { // CHANNELS
          int x2 = x1 + s2o;
          int y2 = y1 + s2p;
          
          int idxPatchData = ji_off + ch;
          int idx2 = ((item * bottomheight + y2+j) * bottomwidth + x2+i) * bottomchannels + ch;
          
          sum[ch_off] += patch_data[idxPatchData] * bottom1[idx2];
        }
      }
    }
    
    __syncthreads();
    
    if(ch_off == 0) {
        float total_sum = 0;
        for(int idx = 0; idx < WARPS_PER_BLOCK*THREADS_PER_WARP; idx++) {
            total_sum += sum[idx];
        }
        const int sumelems = kernel_size*kernel_size*bottomchannels;
        const int index = ((top_channel*topheight + blockIdx.y)*topwidth)+blockIdx.x;
        top[index + item*topcount] = total_sum / (float)sumelems;
    }
  }
  
  // Aggregate  
}

// == Correlation Backward Pass Kernel (For Blob 0)
__global__ void CorrelateDataBackward0(const int nthreads, int num, int item, int topwidth, int topheight, int topchannels,
  int max_displacement, int neighborhood_grid_radius, int neighborhood_grid_width, int kernel_radius, int stride1, int stride2,
  int bottomwidth, int bottomheight, int pbottomwidth, int pbottomheight, int bottomchannels, int bottomcount, int pad_size,
  float *bottom0diff, const float *bottom1, const float *topdiff) 
{
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; index += blockDim.x * gridDim.x){
    int n = index % bottomchannels; //channels
    int l = (index / bottomchannels) % bottomwidth + pad_size; //w-pos
    int m = (index / bottomchannels / bottomwidth) % bottomheight + pad_size; //h-pos

    //Get X,Y ranges and clamp
    // round_off is a trick to enable integer division with ceil, even for negative numbers
    // We use a large offset, for the inner part not to become negative.
    const int round_off = ROUND_OFF;
    const int round_off_s1 = stride1 * round_off;
    
    // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
    int xmin = (l - 2*kernel_radius - max_displacement + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement) / stride1
    int ymin = (m - 2*kernel_radius - max_displacement + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement) / stride1
    
    // Same here:
    int xmax = (l - max_displacement + round_off_s1) / stride1 - round_off; // floor (l - max_displacement) / stride1
    int ymax = (m - max_displacement + round_off_s1) / stride1 - round_off; // floor (m - max_displacement) / stride1
    

    float sum = 0;
    if(xmax>=0 && ymax>=0 && (xmin<=topwidth-1) && (ymin<=topheight-1))
    {
        xmin = max(0,xmin);
        xmax = min(topwidth-1,xmax);

        ymin = max(0,ymin);
        ymax = min(topheight-1,ymax);

        for(int p = -neighborhood_grid_radius; p <= neighborhood_grid_radius; p++) {
          for(int o = -neighborhood_grid_radius; o <= neighborhood_grid_radius; o++) {

            // Get bottom1 data:
            int s2o = stride2 * o;
            int s2p = stride2 * p;
            int idxbot1 = ((item * pbottomheight + (m+s2p)) * pbottomwidth + (l+s2o)) * bottomchannels + n;
            float bot1tmp = bottom1[idxbot1]; // bottom1[l+s2o,m+s2p,n]

            // Index offset for topdiff in following loops:
            int op = (p+neighborhood_grid_radius) * neighborhood_grid_width + (o+neighborhood_grid_radius); // index [o,p]
            int idxopoffset = (item * topchannels + op);

            for(int y = ymin; y <= ymax; y++) {
              for(int x = xmin; x <= xmax; x++) {
                int idxtopdiff = (idxopoffset * topheight + y) * topwidth + x; // topdiff[x,y,o,p]
                sum += topdiff[idxtopdiff] * bot1tmp;
              }
            }
          }
        }
    }
    const int sumelems = (kernel_radius*2+1)*(kernel_radius*2+1)*bottomchannels;
		const int bot0index = ((n * bottomheight) + (m-pad_size)) * bottomwidth + (l-pad_size);
    bottom0diff[bot0index + item*bottomcount] = sum / (float)sumelems;
  }

}



// == Correlation Backward Pass Kernel (For Blob 1)
__global__ void CorrelateDataBackward1(const int nthreads, int num, int item, int topwidth, int topheight, int topchannels,
  int max_displacement, int neighborhood_grid_radius, int neighborhood_grid_width, int kernel_radius, int stride1, int stride2,
  int bottomwidth, int bottomheight, int pbottomwidth, int pbottomheight, int bottomchannels, int bottomcount, int pad_size,
  const float *bottom0, float *bottom1diff, const float *topdiff) 
{
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; index += blockDim.x * gridDim.x){
    //int l = index % bottomwidth + pad_size; //w-pos
    //int m = (index / bottomwidth) % bottomheight + pad_size; //h-pos
    //int n = (index / bottomwidth / bottomheight) % bottomchannels; //channels
    int n = index % bottomchannels; //channels
    int l = (index / bottomchannels) % bottomwidth + pad_size; //w-pos
    int m = (index / bottomchannels / bottomwidth) % bottomheight + pad_size; //h-pos
    
    // round_off is a trick to enable integer division with ceil, even for negative numbers
    // We use a large offset, for the inner part not to become negative.
    const int round_off = ROUND_OFF;
    const int round_off_s1 = stride1 * round_off;
    
    float sum = 0;
    for(int p = -neighborhood_grid_radius; p <= neighborhood_grid_radius; p++) {
      for(int o = -neighborhood_grid_radius; o <= neighborhood_grid_radius; o++) {
        
        int s2o = stride2 * o;
        int s2p = stride2 * p;
        
        //Get X,Y ranges and clamp
        // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
        int xmin = (l - 2*kernel_radius - max_displacement - s2o + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement - s2o) / stride1
        int ymin = (m - 2*kernel_radius - max_displacement - s2p + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement - s2o) / stride1
        
        // Same here:
        int xmax = (l - max_displacement - s2o + round_off_s1) / stride1 - round_off; // floor (l - max_displacement - s2o) / stride1
        int ymax = (m - max_displacement - s2p + round_off_s1) / stride1 - round_off; // floor (m - max_displacement - s2p) / stride1

        if(xmax>=0 && ymax>=0 && (xmin<=topwidth-1) && (ymin<=topheight-1))
        {
            xmin = max(0,xmin);
            xmax = min(topwidth-1,xmax);

            ymin = max(0,ymin);
            ymax = min(topheight-1,ymax);

            // Get bottom0 data:
            int idxbot0 = ((item * pbottomheight + (m-s2p)) * pbottomwidth + (l-s2o)) * bottomchannels + n;
            float bot0tmp = bottom0[idxbot0]; // bottom1[l+s2o,m+s2p,n]

            // Index offset for topdiff in following loops:
            int op = (p+neighborhood_grid_radius) * neighborhood_grid_width + (o+neighborhood_grid_radius); // index [o,p]
            int idxOpOffset = (item * topchannels + op);

            for(int y = ymin; y <= ymax; y++) {
              for(int x = xmin; x <= xmax; x++) {
                int idxtopdiff = (idxOpOffset * topheight + y) * topwidth + x; // topdiff[x,y,o,p]
                sum += topdiff[idxtopdiff] * bot0tmp;
              }
            }
        }
      }
    }
    const int sumelems = (kernel_radius*2+1)*(kernel_radius*2+1)*bottomchannels;
		const int bot1index = ((n * bottomheight) + (m-pad_size)) * bottomwidth + (l-pad_size);
		bottom1diff[bot1index + item*bottomcount] = sum / (float)sumelems;
  }

}

__global__ 
void CorrelateDataSubtract(const int nthreads, int num, int item, int topwidth, int topheight, int topchannels, int topcount,
  int max_displacement, int neighborhood_grid_radius, int neighborhood_grid_width, int kernel_radius, int stride1, int stride2,
  int bottomwidth, int bottomheight, int bottomchannels,
  const float *bottom0, const float *bottom1, float *top) 
{
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; index += blockDim.x * gridDim.x){
    int x = index % topwidth; //w-pos
    int y = (index / topwidth) % topheight; //h-pos
    int c = (index / topwidth / topheight) % topchannels; //channels
        
    // Offset of patch in image 2
    int s2o = (c % neighborhood_grid_width - neighborhood_grid_radius) * stride2;
    int s2p = (c / neighborhood_grid_width - neighborhood_grid_radius) * stride2;
        
    // First (upper left) position of kernel center in current neighborhood in image 1
    int x1 = x*stride1 + kernel_radius + max_displacement;
    int y1 = y*stride1 + kernel_radius + max_displacement;
    
    // Iterate through 3D patch
    float sum = 0;
    for(int j = -kernel_radius; j <= kernel_radius; j++) { // HEIGHT
      for(int i = -kernel_radius; i <= kernel_radius; i++) { // WIDTH
        for(int l = 0; l < bottomchannels; l++) { // CHANNELS
          // Calculate position in image 2
          int x2 = x1 + s2o;
          int y2 = y1 + s2p;

          // Indices in bottom data: (CH=l,W=x2,H=y2,N)
          int idx1 = ((item * bottomheight + y1+j) * bottomwidth + x1+i) * bottomchannels + l;
          int idx2 = ((item * bottomheight + y2+j) * bottomwidth + x2+i) * bottomchannels + l;

          // Do the correlation:
          sum += fabsf(bottom0[idx1] - bottom1[idx2]);
        }
      }
    }
    const int sumelems = (kernel_radius*2+1)*(kernel_radius*2+1)*bottomchannels;
    top[index + item*topcount] = sum / (float)sumelems;
  }

}

// == Correlation Backward Pass Kernel (For Blob 0)
__global__ void CorrelateDataBackward0Subtract(const int nthreads, int num, int item, int topwidth, int topheight, int topchannels,
  int max_displacement, int neighborhood_grid_radius, int neighborhood_grid_width, int kernel_radius, int stride1, int stride2,
  int bottomwidth, int bottomheight, int pbottomwidth, int pbottomheight, int bottomchannels, int bottomcount, int pad_size,
  float *bottom0diff, const float *bottom0, const float *bottom1, const float *topdiff) 
{
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; index += blockDim.x * gridDim.x){
    int l = index % bottomwidth + pad_size; //w-pos
    int m = (index / bottomwidth) % bottomheight + pad_size; //h-pos
    int n = (index / bottomwidth / bottomheight) % bottomchannels; //channels

    //Get X,Y ranges and clamp
    // round_off is a trick to enable integer division with ceil, even for negative numbers
    // We use a large offset, for the inner part not to become negative.
    const int round_off = ROUND_OFF;
    const int round_off_s1 = stride1 * round_off;
    
    // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
    int xmin = (l - 2*kernel_radius - max_displacement + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement) / stride1
    int ymin = (m - 2*kernel_radius - max_displacement + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement) / stride1
    
    // Same here:
    int xmax = (l - max_displacement + round_off_s1) / stride1 - round_off; // floor (l - max_displacement) / stride1
    int ymax = (m - max_displacement + round_off_s1) / stride1 - round_off; // floor (m - max_displacement) / stride1
    

    float sum = 0;
    if(xmax>=0 && ymax>=0 && (xmin<=topwidth-1) && (ymin<=topheight-1))
    {
        xmin = max(0,xmin);
        xmax = min(topwidth-1,xmax);

        ymin = max(0,ymin);
        ymax = min(topheight-1,ymax);

        for(int p = -neighborhood_grid_radius; p <= neighborhood_grid_radius; p++) {
          for(int o = -neighborhood_grid_radius; o <= neighborhood_grid_radius; o++) {

            // Get bottom1 data:
            int s2o = stride2 * o;
            int s2p = stride2 * p;
            int idxbot = ((item * pbottomheight + (m+s2p)) * pbottomwidth + (l+s2o)) * bottomchannels + n;
            float bot0tmp = bottom0[idxbot]; // bottom0[l+s2o,m+s2p,n]
            float bot1tmp = bottom1[idxbot]; // bottom1[l+s2o,m+s2p,n]
            float sign = (bot0tmp >= bot1tmp) ? float(1.0) : float(-1.0);

            // Index offset for topdiff in following loops:
            int op = (p+neighborhood_grid_radius) * neighborhood_grid_width + (o+neighborhood_grid_radius); // index [o,p]
            int idxopoffset = (item * topchannels + op);

            for(int y = ymin; y <= ymax; y++) {
              for(int x = xmin; x <= xmax; x++) {
                int idxtopdiff = (idxopoffset * topheight + y) * topwidth + x; // topdiff[x,y,o,p]
                sum += topdiff[idxtopdiff] * sign;
              }
            }
          }
        }
    }
    const int sumelems = (kernel_radius*2+1)*(kernel_radius*2+1)*bottomchannels;
    bottom0diff[index + item*bottomcount] = sum / (float)sumelems;
  }

}


// == Correlation Backward Pass Kernel (For Blob 1)
__global__ void CorrelateDataBackward1Subtract(const int nthreads, int num, int item, int topwidth, int topheight, int topchannels,
  int max_displacement, int neighborhood_grid_radius, int neighborhood_grid_width, int kernel_radius, int stride1, int stride2,
  int bottomwidth, int bottomheight, int pbottomwidth, int pbottomheight, int bottomchannels, int bottomcount, int pad_size,
  const float *bottom0, const float *bottom1, float *bottom1diff, const float *topdiff) 
{
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; index += blockDim.x * gridDim.x){
    int l = index % bottomwidth + pad_size; //w-pos
    int m = (index / bottomwidth) % bottomheight + pad_size; //h-pos
    int n = (index / bottomwidth / bottomheight) % bottomchannels; //channels
    
    // round_off is a trick to enable integer division with ceil, even for negative numbers
    // We use a large offset, for the inner part not to become negative.
    const int round_off = ROUND_OFF;
    const int round_off_s1 = stride1 * round_off;
    
    float sum = 0;
    for(int p = -neighborhood_grid_radius; p <= neighborhood_grid_radius; p++) {
      for(int o = -neighborhood_grid_radius; o <= neighborhood_grid_radius; o++) {
        
        int s2o = stride2 * o;
        int s2p = stride2 * p;
        
        //Get X,Y ranges and clamp
        // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
        int xmin = (l - 2*kernel_radius - max_displacement - s2o + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement - s2o) / stride1
        int ymin = (m - 2*kernel_radius - max_displacement - s2p + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement - s2o) / stride1
        
        // Same here:
        int xmax = (l - max_displacement - s2o + round_off_s1) / stride1 - round_off; // floor (l - max_displacement - s2o) / stride1
        int ymax = (m - max_displacement - s2p + round_off_s1) / stride1 - round_off; // floor (m - max_displacement - s2p) / stride1

        if(xmax>=0 && ymax>=0 && (xmin<=topwidth-1) && (ymin<=topheight-1))
        {
            xmin = max(0,xmin);
            xmax = min(topwidth-1,xmax);

            ymin = max(0,ymin);
            ymax = min(topheight-1,ymax);

            // Get bottom0 data:
            int idxbot = ((item * pbottomheight + (m-s2p)) * pbottomwidth + (l-s2o)) * bottomchannels + n;
            float bot0tmp = bottom0[idxbot]; // bottom0[l+s2o,m+s2p,n]
            float bot1tmp = bottom1[idxbot]; // bottom1[l+s2o,m+s2p,n]
            float sign = (bot0tmp >= bot1tmp) ? float(-1.0) : float(1.0);

            // Index offset for topdiff in following loops:
            int op = (p+neighborhood_grid_radius) * neighborhood_grid_width + (o+neighborhood_grid_radius); // index [o,p]
            int idxOpOffset = (item * topchannels + op);

            for(int y = ymin; y <= ymax; y++) {
              for(int x = xmin; x <= xmax; x++) {
                int idxtopdiff = (idxOpOffset * topheight + y) * topwidth + x; // topdiff[x,y,o,p]
                sum += topdiff[idxtopdiff] * sign;
              }
            }
        }
      }
    }
    const int sumelems = (kernel_radius*2+1)*(kernel_radius*2+1)*bottomchannels;
    bottom1diff[index + item*bottomcount] = sum / (float)sumelems;
  }

}

int CorrForwardLaucher(
	const float *bot0_flat, const float *bot1_flat, float *rbot1_flat, float *rbot2_flat, float *output_flat, 
	const int bnum, const int bchannels, const int bheight, const int bwidth, const int top_channels, const int top_height, const int top_width,
	const int max_displacement, const int kernel_size, const int pad, const int stride_1, const int stride_2, const int corr_type, 
	const int if_abs, const int kernel_radius, const int border_size, const int neighborhood_grid_radius, 
	const int neighborhood_grid_width,
	cudaStream_t stream){
	//cudaDeviceSynchronize();
	//clock_t before = clock();

	const int bwidthheight = bwidth*bheight;
	const int topcount = top_width*top_height*top_channels;

	dim3 threadsPerBlock(THREADS_PER_WARP * WARPS_PER_BLOCK); 

	int threads_per_block=16;
    dim3 totalBlocksRearr((bwidthheight-1)/threads_per_block+1, bchannels, bnum);
    const int pwidthheight = (bwidth + 2 * pad) * (bheight + 2 * pad);
    
    blob_rearrange_kernel2<<<totalBlocksRearr,threads_per_block>>>
            (bot0_flat,rbot1_flat,bnum,bchannels,bwidth,bheight,bwidthheight,pad,pwidthheight);
    
    blob_rearrange_kernel2<<<totalBlocksRearr,threads_per_block>>>
            (bot1_flat,rbot2_flat,bnum,bchannels,bwidth,bheight,bwidthheight,pad,pwidthheight);

    const int num = bnum;
    const int channels = bchannels;
    const int height = bheight + 2*pad;
    const int width = bwidth + 2*pad;
    
    const int shared_memory_per_block = (kernel_size*kernel_size)*bchannels;

    if(corr_type == 0) { // 0 for multi
        // CorrelationLayer
        int topThreadCount = topcount;
        
        dim3 totalBlocksCorr(top_width, top_height, num);
        
        
        CorrelateData<<<totalBlocksCorr, threadsPerBlock, shared_memory_per_block * sizeof(float)>>>(
            topThreadCount,
            num, top_width, top_height, top_channels, topcount,
            max_displacement, neighborhood_grid_radius, neighborhood_grid_width, kernel_radius, kernel_size,
            stride_1, stride_2,
            width, height, channels,
            rbot1_flat, rbot2_flat, output_flat
            );

        THCudaCheck(cudaGetLastError());
        
    } else if(corr_type == 1) { // 1 for substract
        // CorrelationLayer
        for(int n = 0; n < num; n++) {
            
            int topThreadCount = topcount;
            
            CorrelateDataSubtract<<<(topThreadCount+511)/512, 512>>>(
                topThreadCount,
                num, n, top_width, top_height, top_channels, topcount,
                max_displacement, neighborhood_grid_radius, neighborhood_grid_width, kernel_radius,
                stride_1, stride_2,
                width, height, channels,
                rbot1_flat, rbot2_flat, output_flat
                );

            
            THCudaCheck(cudaGetLastError());
        }
    }

	//cudaDeviceSynchronize();
	//clock_t after = clock();
	//int msec = (after-before)*1000/CLOCKS_PER_SEC;
	//printf("forward elapsed time in ms: %d\n", msec);
	THCudaCheck(cudaGetLastError());
	return 1;

}



int CorrBackwardLaucher(
    const float* output_flat, float* input0_flat, float* input1_flat, const float* rbot1_flat, const float* rbot2_flat, 
    const int num, const int channels, const int height, const int width, const int top_channels, const int top_height, const int top_width,
	const int max_displacement, const int kernel_size, const int pad, const int stride_1, const int stride_2, const int corr_type, 
	const int if_abs, const int kernel_radius, const int border_size, const int neighborhood_grid_radius, 
	const int neighborhood_grid_width,
	cudaStream_t stream){
	//cudaDeviceSynchronize();
	//clock_t before = clock();
	
	const int paddedheight = height + 2*pad;
    const int paddedwidth = width + 2*pad;

    const int bottomcount = channels * height * width;

    int botThreadCount = bottomcount;

    if(corr_type == 0) { // 0 for multi
        
        // == Run kernel Backward 0
        dim3 totalBlocksBackward0(width, height, channels * num); //First dim is fastest
        dim3 threadsPerBlockBackward0(THREADS_PER_WARP * WARPS_PER_BLOCK); 
        const int buffer_size_backw0 = ((int)ceil((float)(2 * kernel_radius) / (float)stride_1) + 1) * top_channels;
       
        // == Run kernel Backward 0 
        for(int n = 0; n < num; n++) {
        //Bottom0:
        CorrelateDataBackward0<<<(botThreadCount+511)/512, 512>>>(
            botThreadCount,
            num, n, top_width, top_height, top_channels,
            max_displacement, neighborhood_grid_radius, neighborhood_grid_width, kernel_radius,
            stride_1, stride_2,
            width, height, paddedwidth, paddedheight, channels, bottomcount, pad,
            input0_flat, rbot2_flat, output_flat
            );
    
        THCudaCheck(cudaGetLastError());
        }
        
        // == Run kernel Backward 1
        for(int n = 0; n < num; n++) {
        CorrelateDataBackward1<<<(botThreadCount+511)/512, 512>>>(
            botThreadCount,
            num, n, top_width, top_height, top_channels,
            max_displacement, neighborhood_grid_radius, neighborhood_grid_width, kernel_radius,
            stride_1, stride_2,
            width, height, paddedwidth, paddedheight, channels, bottomcount, pad,
            rbot1_flat, input1_flat, output_flat
            );
    
        THCudaCheck(cudaGetLastError());
        }
        
    } else if(corr_type == 1) { // 1 for substract
        for(int n = 0; n < num; n++) {
        //Bottom0:
        CorrelateDataBackward0Subtract<<<(botThreadCount+511)/512, 512>>>(
            botThreadCount,
            num, n, top_width, top_height, top_channels,
            max_displacement, neighborhood_grid_radius, neighborhood_grid_width, kernel_radius,
            stride_1, stride_2,
            width, height, paddedwidth, paddedheight, channels, bottomcount, pad,
            input0_flat, rbot1_flat, rbot2_flat, output_flat
            );
    
        THCudaCheck(cudaGetLastError());
        }

        for(int n = 0; n < num; n++) {
        //Bottom0:
        CorrelateDataBackward1Subtract<<<(botThreadCount+511)/512, 512>>>(
            botThreadCount,
            num, n, top_width, top_height, top_channels,
            max_displacement, neighborhood_grid_radius, neighborhood_grid_width, kernel_radius,
            stride_1, stride_2,
            width, height, paddedwidth, paddedheight, channels, bottomcount, pad,
            rbot1_flat, rbot2_flat, input1_flat, output_flat
            );
    
        THCudaCheck(cudaGetLastError());
        }
    }

	//cudaDeviceSynchronize();
	//clock_t after = clock();
	//int msec = (after-before)*1000/CLOCKS_PER_SEC;
	//printf("backward elapsed time in ms: %d\n", msec);
	THCudaCheck(cudaGetLastError());
	return 1;
}

#ifdef __cplusplus
}
#endif
