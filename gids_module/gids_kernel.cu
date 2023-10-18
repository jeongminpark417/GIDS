

template <typename T = float>
__global__ void read_feature_kernel(array_d_t<T> *dr, T *out_tensor_ptr,
                                    int64_t *index_ptr, int dim,
                                    int64_t num_idx, int cache_dim, uint64_t key_off) {
 uint64_t bid = blockIdx.x;
  int num_warps = blockDim.x / 32;
  int warp_id = threadIdx.x / 32;
  int idx_idx = bid * num_warps + warp_id;
  if (idx_idx < num_idx) {
 	    bam_ptr<T> ptr(dr);

       	  uint64_t row_index = index_ptr[idx_idx] + key_off;
      	uint64_t tid = threadIdx.x % 32;


    for (; tid < dim; tid += 32) {
	    T temp = ptr[(row_index) * cache_dim + tid];
	    out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = temp;
    }
  }
}

template <typename T = float>
__global__ void read_feature_kernel_with_cpu_backing_memory(array_d_t<T> *dr, range_d_t<T> *range, T *out_tensor_ptr,
                                    int64_t *index_ptr, int dim,
                                    int64_t num_idx, int cache_dim, GIDS_CPU_buffer<T> CPU_buffer, bool cpu_seq, unsigned int* d_cpu_access, uint64_t key_off) {

  uint64_t bid = blockIdx.x;

  int num_warps = blockDim.x / 32;
  int warp_id = threadIdx.x / 32;
  int idx_idx = bid * num_warps + warp_id;
  if (idx_idx < num_idx) {
 	    bam_ptr<T> ptr(dr);

      uint64_t row_index = index_ptr[idx_idx] + key_off;
      uint64_t tid = threadIdx.x % 32;

      uint32_t cpu_off = range -> get_cpu_offset(row_index);


      if(cpu_seq){
        if(row_index < CPU_buffer.cpu_buffer_len){
          if(tid == 0)
            atomicAdd(d_cpu_access, 1);
          for (; tid < dim; tid += 32) {
            T temp = CPU_buffer.device_cpu_buffer[(row_index) * cache_dim + tid];
            out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = temp;
            }
        }

        else{
        for (; tid < dim; tid += 32) {
          T temp = ptr[(row_index) * cache_dim + tid];
          out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = temp;
        }
      }
      }
      else{
        if((cpu_off & 0x1) == 1){
          if(tid == 0)
            atomicAdd(d_cpu_access, 1);

            for (; tid < dim; tid += 32) {
              T temp = CPU_buffer.device_cpu_buffer[(cpu_off >> 1) * cache_dim + tid];
              out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = temp;
            }
        }

        else{
          for (; tid < dim; tid += 32) {
            T temp = ptr[(row_index) * cache_dim + tid];
            out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = temp;
          }
        }
      }
  }
}


template <typename T = float>
__global__ void set_cpu_buffer_kernel(range_d_t<T> *d_range, uint64_t* idx_ptr, int num, uint32_t pageSize) {
  
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if(idx <  num){
    d_range -> set_cpu_buffer(idx_ptr[idx], idx );
  }
}


template <typename T = float>
__global__
void set_window_buffering_kernel(array_d_t<T>* dr, uint64_t *index_ptr, uint64_t page_size, int hash_off){
	bam_ptr<T> ptr(dr);
	if(threadIdx.x == 0){
		uint64_t page_idx = index_ptr[blockIdx.x] + hash_off;
		ptr.set_window_buffer_counter(page_idx * page_size/sizeof(T), 1);
	}
}

template <typename T = float>
__global__ void read_kernel(array_d_t<T> *dr,
                                    uint64_t num, uint64_t offset) {
      bam_ptr<T> ptr(dr);
     if(threadIdx.x == 0 && blockIdx.x == 0){
        for(uint64_t i = 0; i < num; i++){
              if(i == 0) printf("idx: %llu type size:%i \n", offset,  (int) sizeof(T));
             // T temp = ptr[i + offset];
              printf("read data: %llu\n",  (unsigned long long) ptr[i + offset]);
             // printf("float read data: %f\n", temp);

        }
     }                           
}

template <typename T = float>
__global__ void write_feature_kernel(Controller** ctrls, page_cache_d_t* pc, array_d_t<T> *dr, T* in_tensor_ptr,
                                    uint64_t num, uint64_t offset) {

    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < num){
      bam_ptr<T> ptr(dr);
      ptr[idx + offset] = in_tensor_ptr[idx];
    }
}


