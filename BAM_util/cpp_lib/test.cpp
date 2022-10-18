#include <cstdint>
#include "cuda_lib.h"

int main(){
	float a = 1;
	int32_t idx = 0;
	int64_t i_ptr = 100000;
   h_cuda_write(i_ptr,  idx, a);
   return 0;
}
