
#include "cuda_lib.h"

__global__ 
void cuda_write(float* ptr, int32_t idx, float a){
	if(threadIdx.x == 0)
		ptr[idx] = a;

}

void Direct_Loader::hi(){
	printf("hi\n");
}

void Direct_Loader::h_cuda_write(int64_t i_ptr, int32_t idx, float a)
{

	float* ptr = (float*) i_ptr;
	cuda_write<<<1,32>>>(ptr,idx,a);
	cudaDeviceSynchronize();
}
