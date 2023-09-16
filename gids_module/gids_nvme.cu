#include <pybind11/pybind11.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <buffer.h>
#include <cuda.h>
#include <fcntl.h>
#include <nvm_admin.h>
#include <nvm_cmd.h>
#include <nvm_ctrl.h>
#include <nvm_error.h>
#include <nvm_io.h>
#include <nvm_parallel_queue.h>
#include <nvm_queue.h>
#include <nvm_types.h>
#include <nvm_util.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>
#include <util.h>

#include <ctrl.h>
#include <event.h>
#include <page_cache.h>
#include <queue.h>
#include <stdio.h>
#include <vector>

#include <bam_nvme.h>
//#include <bafs_ptr.h>

template<typename T >
void  cpu_aggregate_kernel(uint64_t* location_arrary, uint64_t list_len,  T * pinned_queue_ptr, uint64_t q_size,  T * dst_buf, int32_t head_ptr, uint32_t wb_depth, uint64_t dim){

  #define encode_mask 0x0000FFFFFFFFFFFF

  #pragma omp parallel
  #pragma omp for
  for(uint64_t i = 0; i < list_len; i++){

    //(reuse_sec & (0x0000FFFFFFFFFFFF));

    uint64_t encoded_src = location_arrary[i];

    uint64_t head_idx = ((encoded_src >> 48) + head_ptr) % wb_depth;
    uint64_t q_idx = (encoded_src & encode_mask);

    //std::cout << "head : " << head_idx << " idx: " << q_idx << std::endl;
    T* src_ptr = pinned_queue_ptr + head_idx * q_size + q_idx;
    T* dst_ptr = dst_buf + i * dim;
    //T* dst_ptr = dst_buf;
    std::memcpy(dst_ptr, src_ptr, dim * sizeof(T)); // OK
  }
};




typedef std::chrono::high_resolution_clock Clock;

template <typename T = float>
__global__ void read_feature_kernel(array_d_t<T> *dr, T *out_tensor_ptr,
                                    int64_t *index_ptr, int dim,
                                    int64_t num_idx, int cache_dim) {
 uint64_t bid = blockIdx.x;
  int num_warps = blockDim.x / 32;
  int warp_id = threadIdx.x / 32;
  int idx_idx = bid * num_warps + warp_id;
  if (idx_idx < num_idx) {
 	    bam_ptr<T> ptr(dr);

       	  uint64_t row_index = index_ptr[idx_idx];
      	uint64_t tid = threadIdx.x % 32;


    for (; tid < dim; tid += 32) {
	    T temp = ptr[(row_index) * cache_dim + tid];
	    out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = temp;
    }
  }
}


template <typename T = float>
__global__ void read_feature_kernel_with_wb (array_d_t<T> *dr, T *out_tensor_ptr,
                                    int64_t *index_ptr, int dim,
                                    uint64_t num_idx, int cache_dim,
                                    uint32_t* wb_queue_counter,  
                                    uint32_t  wb_depth,  T* queue_ptr, 
                                    uint64_t* wb_id_array, uint32_t q_depth,
                                    uint32_t* node_flag_ptr, T* node_ptr, 
                                    uint8_t time_step, uint32_t head_ptr,
                                    unsigned int* hit_counter) {
    
  uint64_t bid = blockIdx.x;
  int num_warps = blockDim.x / 32;
  int warp_id = threadIdx.x / 32;
  int idx_idx = bid * num_warps + warp_id;
  if (idx_idx < num_idx) {
	   uint32_t fetch_idx = node_flag_ptr[idx_idx] ;

      	uint64_t tid = threadIdx.x % 32;
 	            //already prefetched
        if(fetch_idx != 0) {
          //atomicAdd(hit_counter, (unsigned int)1);
	      		
          for(; tid < dim; tid += 32){
            out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = node_ptr[dim * fetch_idx + tid];
      //			out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = node_ptr[0];
          }
        }

    	else{
	wb_bam_ptr<T> ptr(dr);
        ptr.set_wb(wb_queue_counter, wb_depth, queue_ptr, wb_id_array, q_depth);
        //ptr.set_time(head_ptr, time_step);
        ptr.set_time( time_step, head_ptr);

       	uint64_t row_index = index_ptr[idx_idx];


		for (; tid < dim; tid += 32) {
      
      	    T temp = ptr[(row_index ) * cache_dim + tid];
	    out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = temp;
    
		}
	}
  }

}


template <typename T = float>
__global__
void set_prefetching_kernel(array_d_t<T>* dr, uint64_t *index_ptr, uint8_t* p_val_ptr, uint64_t page_size){
	bam_ptr<T> ptr(dr);
	if(threadIdx.x == 0){
		uint64_t page_idx = index_ptr[blockIdx.x];
		uint8_t p_val = p_val_ptr[blockIdx.x];
		ptr.set_prefetch_val(page_idx * page_size/sizeof(T), p_val);
	}
}

template <typename T = float>
__global__
void set_window_buffering_kernel(array_d_t<T>* dr, uint64_t *index_ptr, uint8_t* p_val_ptr, uint64_t page_size){
	bam_ptr<T> ptr(dr);
	if(threadIdx.x == 0){
		uint64_t page_idx = index_ptr[blockIdx.x];
		uint8_t p_val = p_val_ptr[blockIdx.x];
		ptr.set_window_buffer_counter(page_idx * page_size/sizeof(T), p_val);
	}
}

//new fn

template <typename T = float>
__global__
void update_wb_counters(array_d_t<T> *dr,uint64_t** batch_arrays, uint64_t* batch_size_array, uint64_t wb_size, uint8_t time_step){
    //x dim: each node in batch
    //y dim: depth of wb
    uint32_t cur_iter = blockIdx.y;


    int32_t cur_batch_node = blockIdx.x * blockDim.x + threadIdx.x;

    const uint64_t my_batch_len = batch_size_array[cur_iter];
    wb_bam_ptr<T> ptr(dr);

    if(cur_batch_node < my_batch_len){
	    uint64_t* cur_batch = batch_arrays[cur_iter];

	 uint64_t cur_node = cur_batch[cur_batch_node];
	 ptr.update_wb(cur_node,  cur_iter , cur_batch_node);
    }
}

template <typename T = float>
__global__
void update_wb_counters_list(array_d_t<T> *dr,uint64_t** batch_arrays, uint64_t* batch_size_array, uint64_t wb_size, uint8_t time_step){
    //x dim: each node in batch
    //y dim: depth of wb
    uint64_t cur_iter = blockIdx.y;


    int32_t cur_batch_node = blockIdx.x * blockDim.x + threadIdx.x;

    const uint64_t my_batch_len = batch_size_array[cur_iter];
    wb_bam_ptr<T> ptr(dr);
    if(cur_batch_node < my_batch_len){
	    uint64_t* cur_batch = batch_arrays[cur_iter];
	    uint64_t cur_node = cur_batch[cur_batch_node];
	    ptr.update_wb_list(cur_node,  cur_iter , cur_batch_node);
    }
}

#define BSIZE 128

template <typename T = float>
__global__
void update_wb_counters_test(array_d_t<T> *dr, range_d_t<TYPE> *d_range, uint64_t** batch_arrays, uint64_t* batch_size_array, uint64_t wb_size, uint8_t time_step){
    //x dim: each node in batch
    //y dim: depth of wb
 

	uint32_t cur_iter = blockIdx.y;


    int32_t cur_batch_node = blockIdx.x * blockDim.x + threadIdx.x;

    const uint64_t my_batch_len = batch_size_array[cur_iter];
    wb_bam_ptr<T> ptr(dr);

    if(cur_batch_node < my_batch_len){
            uint64_t* cur_batch = batch_arrays[cur_iter];

         uint64_t cur_node = cur_batch[cur_batch_node];
    	uint64_t page_trans;
 	d_range -> wb_check_page(cur_node, page_trans);
//	if(page_trans == 1) printf("trans 1\n");
	// ptr.update_wb_test(cur_node,  cur_iter , cur_batch_node);
	 //     ptr.update_wb_test(cur_node,  cur_iter , cur_batch_node);
    }
    
}


template <typename T = float>
__global__
void update_wb_counters_seq(array_d_t<T> *dr,uint64_t** batch_arrays, uint64_t* batch_size_array, uint64_t wb_size, uint8_t cur_iter){

    uint64_t* cur_batch = batch_arrays[cur_iter];

    int32_t cur_batch_node = threadIdx.x;

    uint64_t my_batch_len = batch_size_array[cur_iter];
    wb_bam_ptr<T> ptr(dr);

    for(uint32_t i = cur_batch_node; i < my_batch_len; i+=blockDim.x){
        uint64_t cur_node = cur_batch[i];
        ptr.update_wb(cur_node,  cur_iter + 0);
    }
}



template <typename T = float>
__global__ void bam_pin_cache(array_d_t<T>* dr, int64_t *index_ptr, int dim,int64_t num_idx, uint64_t page_size){
	
	bam_ptr<T> ptr(dr);
	int64_t page_index = index_ptr[blockIdx.x];
	uint64_t pin_idx = page_index * page_size / sizeof(T) + threadIdx.x;
	dr -> pin_memory(pin_idx);
}




template <typename T = uint64_t>
__global__ void sequential_access_kernel(array_d_t<T> *dr, uint64_t n_reqs,
                                         unsigned long long *req_count,
                                         uint64_t reqs_per_thread) {

  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < n_reqs) {
    for (size_t i = 0; i < reqs_per_thread; i++)
      req_count += (uint64_t)(*dr)[(tid)];

    T test = (*dr)[(tid)];
    float *f_ptr = (float *)(&test);

    // printf("tid: %i f1:%f \n", tid, test);
  }
}
void BAM_Feature_Store::mgc_init_controllers(int ps, uint64_t read_off, uint64_t cache_size, uint64_t num_ele = 100, uint64_t ctrl_idx= 0, bool cpu_cache = false, uint64_t cpu_page_num = 0) {

  printf(" mgc init starts gpu idx:%i\n", (int) ctrl_idx);
  uint32_t cudaDevice = ctrl_idx;
  uint64_t cudaDeviceId = ctrl_idx;

  cudaSetDevice(cudaDevice);

  numElems = num_ele;
  read_offset = read_off;
  n_ctrls = 1;

  ctrls.push_back(new Controller(ctrls_paths[ctrl_idx], nvmNamespace, cudaDevice,
                                   queueDepth, numQueues));
  printf("controllers are initalized\n");
  uint64_t b_size = blkSize;
  uint64_t g_size = (numThreads + b_size - 1) / b_size;
  // uint64_t g_size = (numThreads) / b_size;
  uint64_t n_threads = b_size * g_size;

  pageSize = ps;
  uint64_t page_size = pageSize;
  uint64_t n_pages =  cache_size * 1024LL*1024*1024/page_size;
  numPages = n_pages;

  uint64_t total_cache_size = (page_size * n_pages);

  std::cout << "n pages: " << n_pages <<std::endl;
  std::cout << "page size: " << pageSize << std::endl;
  std::cout << "num elements: " << numElems << std::endl;

  h_pc = new page_cache_t(page_size, n_pages, cudaDevice, ctrls[0][0], (uint64_t)64, ctrls);


  page_cache_t *d_pc = (page_cache_t *)(h_pc->d_pc_ptr);
  uint64_t t_size = numElems * sizeof(TYPE);

  std::cout << "numElems: " << numElems << std::endl;
  printf("numElems: %llu t_size:%llu page_size: %llu\n",(uint64_t)numElems,(uint64_t)(t_size),  (uint64_t)page_size);
  h_range = new range_t<TYPE>((uint64_t)0, (uint64_t)numElems, (uint64_t)read_off,
                              (uint64_t)(t_size / page_size), (uint64_t)0,
                              (uint64_t)page_size, h_pc, cudaDevice, 
			      REPLICATE
			      //STRIPE
			      );

  range_d_t<TYPE> *d_range = h_range->d_range_ptr;

  // std::vector<range_t<TYPE>*> vr(1);
  vr.push_back(nullptr);
  vr[0] = h_range;

  //  size_t cpu_cache_size = cpu_page_num * page_size;
   a = new array_t<TYPE>(numElems, 0, vr, cudaDevice);


  //set up wb
 
  cuda_err_chk(cudaMalloc(&wb_queue_counter, sizeof(uint32_t) * wb_depth));
  cuda_err_chk(cudaMalloc(&wb_id_array, sizeof(uint32_t) * wb_depth * wb_queue_depth));

  cuda_err_chk(cudaHostAlloc((TYPE **)&host_wb_queue_ptr, page_size * wb_depth * wb_queue_depth, cudaHostAllocMapped));
  cudaHostGetDevicePointer((TYPE **)&wb_queue_ptr, (TYPE *)host_wb_queue_ptr, 0);

  cuda_err_chk(cudaHostAlloc((uint32_t **)&h_wb_id_array, sizeof(uint32_t) * wb_depth * wb_queue_depth, cudaHostAllocMapped));
  cudaHostGetDevicePointer((uint32_t **)&wb_id_array, (uint32_t *)h_wb_id_array, 0);

  printf("init done\n");

  return;
}

// void BAM_Feature_Store::init_controllers(const char *const ctrls_paths[]) {
void BAM_Feature_Store::init_controllers(int ps, uint64_t read_off, uint64_t cache_size, uint64_t num_ele = 100, uint64_t num_ssd = 1, uint32_t wb_size = 4, uint64_t wb_queue_size = 131072,
bool cpu_agg = false, int32_t cpu_agg_q_depth = 0) {

  printf("init starts num ssd:%llu\n", num_ssd);
  numElems = num_ele;
  read_offset = read_off;
  n_ctrls = num_ssd;
  wb_depth = wb_size;
  pageSize = ps;
  dim = ps / sizeof(TYPE);
  wb_queue_depth = wb_queue_size;
  cpu_agg_queue_depth =cpu_agg_q_depth;
  total_access = 0; 
  prefetch_count = 0;
  memcpy_count = 0;
  overlap = 0;

  size_t free_mem, total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);
  size_t used_mem = total_mem - free_mem;
    printf("Used GPU Memory: %lu bytes\n", used_mem);
    printf("Free GPU Memory: %lu bytes\n", free_mem);
    printf("Total GPU Memory: %lu bytes\n", total_mem);



  for (size_t i = 0; i < num_ssd; i++) {
    ctrls.push_back(new Controller(ctrls_paths[i], nvmNamespace, cudaDevice,
                                   queueDepth, numQueues));
  }
  printf("controllers are initalized\n");
    cudaMemGetInfo(&free_mem, &total_mem);
   used_mem = total_mem - free_mem;

  printf("Used GPU Memory: %lu MB\n", used_mem / (1024LL*1024));
    printf("Free GPU Memory: %lu MB\n", free_mem / (1024LL*1024));
    printf("Total GPU Memory: %lu MB\n", total_mem / (1024LL*1024));

  uint64_t b_size = blkSize;
  uint64_t g_size = (numThreads + b_size - 1) / b_size;
  // uint64_t g_size = (numThreads) / b_size;
  uint64_t n_threads = b_size * g_size;

  uint64_t page_size = pageSize;
  uint64_t n_pages = cache_size * 1024LL*1024*1024/page_size;
  numPages = n_pages;

  //FIX
  //n_pages = cache_size;
  

  std::cout << "n pages: " << n_pages <<std::endl;
  std::cout << "page size: " << pageSize << std::endl;
  std::cout << "num elements: " << numElems << std::endl;
  std::cout << "window buffer size: " << wb_size << std::endl;
  std::cout << "window queue size: " << wb_queue_depth << std::endl;

  cpu_agg_flag = cpu_agg;
  h_pc = new page_cache_t(page_size, n_pages, cudaDevice, ctrls[0][0],
                          (uint64_t)64, ctrls, wb_depth, cpu_agg, cpu_agg_q_depth);


  page_cache_t *d_pc = (page_cache_t *)(h_pc->d_pc_ptr);
  uint64_t t_size = numElems * sizeof(TYPE);

  printf("create h_range\n");
  std::cout << "numElems: " << numElems << std::endl;
  printf("numElems: %llu t_size:%llu page_size: %llu\n",(uint64_t)numElems,(uint64_t)(t_size),  (uint64_t)page_size);
  h_range = new range_t<TYPE>((uint64_t)0, (uint64_t)numElems, (uint64_t)read_off,
                              (uint64_t)(t_size / page_size), (uint64_t)0,
                              (uint64_t)page_size, h_pc, cudaDevice, 
			      REPLICATE
			      //STRIPE
			      );

  printf("h rnage created\n");
  
  d_range = (range_d_t<TYPE> *)h_range->d_range_ptr;

  // std::vector<range_t<TYPE>*> vr(1);
  vr.push_back(nullptr);
  vr[0] = h_range;

  cudaMemGetInfo(&free_mem, &total_mem);
   used_mem = total_mem - free_mem;

  printf("Used GPU Memory: %lu MB\n", used_mem / (1024LL*1024));
    printf("Free GPU Memory: %lu MB\n", free_mem / (1024LL*1024));
    printf("Total GPU Memory: %lu MB\n", total_mem / (1024LL*1024));



  /*
  array_t<TYPE> a2(numElems, 0, vr, cudaDevice);
  a = std::move(a2);
*/
  printf("array_t crate\n");
//  size_t cpu_cache_size = cpu_page_num * page_size;
   a = new array_t<TYPE>(numElems, 0, vr, cudaDevice);


  printf("init done wb_depth: %i\n", (int) wb_depth);

  //set up wb
  cudaMemGetInfo(&free_mem, &total_mem);
   used_mem = total_mem - free_mem;
 
  printf("Used GPU Memory: %lu MB\n", used_mem / (1024LL*1024));
  printf("Free GPU Memory: %lu MB\n", free_mem / (1024LL*1024));
  printf("Total GPU Memory: %lu MB\n", total_mem / (1024LL*1024));

  //cpu_agg_buffer = (TYPE*) malloc(page_size * cpu_agg_queue_depth);
  cuda_err_chk(cudaHostAlloc((TYPE **)&cpu_agg_buffer, page_size * cpu_agg_queue_depth, cudaHostAllocMapped));

  if(cpu_agg_flag){
    cuda_err_chk(cudaMalloc(&d_agg_buffer, page_size * cpu_agg_queue_depth));
    cuda_err_chk(cudaMalloc(&d_agg_loc, sizeof(uint64_t) * cpu_agg_queue_depth));
  }

  cuda_err_chk(cudaMalloc(&wb_queue_counter, sizeof(uint32_t) * wb_depth));
  cuda_err_chk(cudaMalloc(&wb_id_array, sizeof(uint64_t) * wb_depth * wb_queue_depth ));

  cuda_err_chk(cudaHostAlloc((TYPE **)&host_wb_queue_ptr, page_size * wb_depth * wb_queue_depth, cudaHostAllocMapped));
  cudaHostGetDevicePointer((TYPE **)&wb_queue_ptr, (TYPE *)host_wb_queue_ptr, 0);

  cuda_err_chk(cudaHostAlloc((uint64_t **)&h_wb_id_array, sizeof(uint64_t) * wb_depth * wb_queue_depth , cudaHostAllocMapped));
  cudaHostGetDevicePointer((uint64_t **)&wb_id_array, (uint64_t *)h_wb_id_array, 0);

  transfer_count_ptr = (uint32_t*) malloc(sizeof(uint32_t));
  memcpy_count_ptr = (uint32_t*) malloc(sizeof(uint32_t));

  cudaMalloc(&d_batch_array_ptr, sizeof(uint32_t*) * wb_depth);

  cpu_agg_ptr = (TYPE*) malloc (page_size * 10000);
  

  cudaDeviceGetStreamPriorityRange(&low_priority, &high_priority);

  head_ptr = 0;
  time_step = 0;

	printf("Set up WB done\n"); 
  cudaMemGetInfo(&free_mem, &total_mem);
   used_mem = total_mem - free_mem;
   printf("Used GPU Memory: %lu MB\n", used_mem / (1024LL*1024));
    printf("Free GPU Memory: %lu MB\n", free_mem / (1024LL*1024));
    printf("Total GPU Memory: %lu MB\n", total_mem / (1024LL*1024));
  
  return;
}

void  BAM_Feature_Store::set_prefetching(uint64_t id_idx, uint64_t prefetch_idx, int64_t num_pages){
	 uint64_t* idx_ptr = (uint64_t*) id_idx;
	 uint8_t* prefetch_ptr = (uint8_t*) prefetch_idx;
	 uint64_t page_size = pageSize;
	 set_prefetching_kernel<TYPE><<<num_pages, 32>>>(a->d_array_ptr,idx_ptr, prefetch_ptr, page_size);
	 cuda_err_chk(cudaDeviceSynchronize())
}

void  BAM_Feature_Store::set_window_buffering(uint64_t id_idx, uint64_t prefetch_idx, int64_t num_pages){
	 uint64_t* idx_ptr = (uint64_t*) id_idx;
	 uint8_t* prefetch_ptr = (uint8_t*) prefetch_idx;
	 uint64_t page_size = pageSize;
	 set_window_buffering_kernel<TYPE><<<num_pages, 32>>>(a->d_array_ptr,idx_ptr, prefetch_ptr, page_size);
	 cuda_err_chk(cudaDeviceSynchronize())
}

//new fn
template <typename T = float>
__global__ 
void flush_wb_counter(array_d_t<T> *dr, uint32_t num_cl){

    uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid < num_cl){
        wb_bam_ptr<T> ptr(dr);
        ptr.flush_wb_counter(tid);
    }

}

template <typename T = float>
__global__ 
void count_mask_kernel(array_d_t<T> *dr, uint32_t num_cl, uint64_t* counter){

    uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid < num_cl){
        wb_bam_ptr<T> ptr(dr);
        ptr.count_mask(tid, counter);
    }

}


void  BAM_Feature_Store::set_wb_counter(uint64_t batch_array_idx, uint64_t batch_size_idx, uint32_t max_batch_size){
  //printf("set wb counter: %lu\n", (unsigned long) time_step);
  cudaStreamCreateWithPriority (&wb_stream,cudaStreamNonBlocking, low_priority);

  //if(time_step == 1){
  //   time_step = 0;
  //   return;
  // }
  // time_step++;

	uint32_t num_g = (numPages + 127) / 128;
  flush_wb_counter<TYPE><<<num_g, 128, 0 , wb_stream>>>(a->d_array_ptr, (uint32_t)numPages);

  uint64_t** batch_array_ptr = (uint64_t**) batch_array_idx;
  uint64_t* batch_size_ptr = (uint64_t*) batch_size_idx;
  uint64_t page_size = pageSize;

  dim3 b_block(1024, 1, 1);
  uint32_t g_x = (max_batch_size + 1023)/1024;
  dim3 g_block(g_x, wb_depth,1);
    
  auto t1 = Clock::now(); 
  cudaMemcpyAsync(d_batch_array_ptr, batch_array_ptr,  sizeof(uint32_t*) * wb_depth, cudaMemcpyHostToDevice, wb_stream);
  
  update_wb_counters<TYPE><<<g_block, b_block, 0, wb_stream>>>(a->d_array_ptr,batch_array_ptr, batch_size_ptr, wb_depth, time_step);

  auto t2 = Clock::now();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1); // Microsecond (as int)
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1); // Microsecond (as int)
  auto ms_fractional = static_cast<float>(us.count()) / 1000; // Milliseconds (as float)


  set_wb_time += ms_fractional;

  //   cuda_err_chk(cudaDeviceSynchronize());
  // printf("wb done\n");

}


void  BAM_Feature_Store::set_wb_counter_list(uint64_t batch_array_idx, uint64_t batch_size_idx, uint32_t max_batch_size){
  //printf("set wb counter: %lu\n", (unsigned long) time_step);
  cudaStreamCreateWithPriority (&wb_stream,cudaStreamNonBlocking, low_priority);
  // uint64_t* counter;
  // cudaMalloc(&counter, sizeof(uint64_t));

	uint32_t num_g = (numPages + 127) / 128;
  flush_wb_counter<TYPE><<<num_g, 128, 0 , wb_stream>>>(a->d_array_ptr, (uint32_t)numPages);

  uint64_t** batch_array_ptr = (uint64_t**) batch_array_idx;
  uint64_t* batch_size_ptr = (uint64_t*) batch_size_idx;
  uint64_t page_size = pageSize;

  dim3 b_block(1024, 1, 1);
  max_batch_size = 10 * 5 * 5 * 2048;
  uint32_t g_x = (max_batch_size + 1023)/1024;
  dim3 g_block(g_x, wb_depth,1);
    
  auto t1 = Clock::now(); 
  cudaMemcpyAsync(d_batch_array_ptr, batch_array_ptr,  sizeof(uint32_t*) * wb_depth, cudaMemcpyHostToDevice, wb_stream);
  //printf("max batch size: %llu\n", (unsigned long long) max_batch_size);
  update_wb_counters_list<TYPE><<<g_block, b_block, 0, wb_stream>>>(a->d_array_ptr,batch_array_ptr, batch_size_ptr, wb_depth, time_step);

 	// cuda_err_chk(cudaDeviceSynchronize())

  // count_mask_kernel<TYPE><<<num_g, 128, 0 , wb_stream>>>(a->d_array_ptr, (uint32_t)numPages, counter);

	// cuda_err_chk(cudaDeviceSynchronize())

  // uint64_t h_counter;
  // cudaMemcpy(&h_counter, counter, sizeof(uint64_t), cudaMemcpyDeviceToHost);
  // printf("h counter:%llu\n", (unsigned long long)h_counter);

  //std::cout << "h counter: " << (unsigned long long)h_counter << std::endl;

  auto t2 = Clock::now();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1); // Microsecond (as int)
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1); // Microsecond (as int)
  auto ms_fractional = static_cast<float>(us.count()) / 1000; // Milliseconds (as float)
  set_wb_time += ms_fractional; 
  
}

void BAM_Feature_Store::print_stats(){
  std::cout << "print stats: ";
  h_pc->print_reset_stats();
  std::cout << std::endl;

  std::cout << "print array reset: ";
  a->print_reset_stats();
  std::cout << std::endl;

  for(int i = 0; i < n_ctrls; i++){
 	std::cout << "print ctrl reset " << i << ": ";
  	(ctrls[i])->print_reset_stats();
  	std::cout << std::endl;

  }
 
  std::cout << "Kernel Time: \t " << kernel_time << std::endl;
  kernel_time = 0;
  std::cout << "Fill Batch Kernel Time: \t " << fill_batch_time << std::endl;
  fill_batch_time = 0;
  std::cout << "Set WB Counter Kernel Time: \t " << set_wb_time << std::endl;
  set_wb_time = 0;
  std::cout << "Flush Kernel Time: \t " << flush_time << std::endl;
  flush_time = 0;

  std::cout << "Total Access: \t " << total_access << std::endl;
  std::cout << "Prefetch Count: \t " << prefetch_count << std::endl;
  std::cout << "Memcpy Count: \t " << memcpy_count << std::endl;
  std::cout << "overlap Count: \t " << overlap << std::endl;

  total_access = 0;
  prefetch_count = 0;
  memcpy_count = 0;
  overlap = 0;
}


void BAM_Feature_Store::pin_memory(uint64_t i_index_ptr, int64_t num_pin_page, int dim){


  int64_t* index_ptr = (int64_t*) i_index_ptr;
  uint64_t page_size = pageSize;

  std::cout << "Pinning Pages\n";

  bam_pin_cache<TYPE><<<num_pin_page,32>>>(a->d_array_ptr, index_ptr, dim, num_pin_page, page_size);
  cuda_err_chk(cudaDeviceSynchronize());
}

void BAM_Feature_Store::init_backing_memory(size_t memory_size){

	float* h_mem_ptr;
	cudaHostAlloc((void**)&h_mem_ptr, memory_size, cudaHostAllocMapped);

	float* d_mem_ptr;
	cudaHostGetDevicePointer((void**)&d_mem_ptr, h_mem_ptr, 0);

	h_buf_ptr = h_mem_ptr;
	d_buf_ptr = d_mem_ptr;
}


void BAM_Feature_Store::create_streams(int num_streams){
	int lp, hp;
	cudaDeviceGetStreamPriorityRange(&lp, &hp);
	printf("lp: %i hp:%i\n", lp, hp);
	for (int i = 0; i < num_streams;i ++){
		cudaStreamCreateWithPriority(&(stream_array[i]), cudaStreamNonBlocking, lp+1);
	}
}

void BAM_Feature_Store::sync_streams(int num_streams){

	for(int i =0 ; i < num_streams; i++){
		cudaStreamSynchronize(stream_array[i]);
	}

	 cudaStreamSynchronize(stream_array[7]);

}

void BAM_Feature_Store::fetch_from_backing_memory(uint64_t i_device_ptr, uint64_t i_batch_idx_ptr, uint64_t i_backing_idx_ptr,  int batch_size, int cl_size, int num_transfer_cl){
	TYPE *device_ptr = (TYPE *) i_device_ptr;
	int64_t* batch_idx_ptr = (int64_t*) i_batch_idx_ptr;
	int64_t* backing_idx_ptr = (int64_t*) i_backing_idx_ptr;
	
	for(int i = 0; i < num_transfer_cl; i++){
		printf("i:%i\n", i);
		int64_t cur_cl_id = backing_idx_ptr[i];
		for(int j = 0; j < batch_size; j++){
			printf("j:%i\n",j);
			if(batch_idx_ptr[j] = cur_cl_id){
				printf("find i:%i j:%i\n", i, j);
				cudaMemcpy(device_ptr + cl_size/sizeof(4) * j, h_buf_ptr + i * cl_size/sizeof(4), cl_size/sizeof(4), cudaMemcpyHostToDevice);
				break;
			}
		}
	}
	cudaDeviceSynchronize();
}

void BAM_Feature_Store::fetch_from_backing_memory_chunk(uint64_t i_device_ptr, uint64_t cl_size, int stream_id){
	TYPE *device_ptr = (TYPE *) i_device_ptr;
 //  	 cudaMemcpy(device_ptr, h_buf_ptr, cl_size, cudaMemcpyHostToDevice);	
 	 cudaMemcpyAsync(device_ptr, h_buf_ptr, cl_size, cudaMemcpyHostToDevice, stream_array[stream_id]);	
}


void BAM_Feature_Store::prefetch_from_victim_queue(uint64_t i_feature_ptr, uint64_t i_node_id_ptr, int stream_id){
  
 // printf("prefetch_from_victim_queue\n");
  //  printf("head_ptr: %lu transfer count: %lu\n", (unsigned long) head_ptr, (unsigned long)(transfer_count_ptr[0]));
  cudaStreamCreateWithPriority (&transfer_stream,cudaStreamNonBlocking, low_priority);
  cudaStreamCreateWithPriority (&transfer_stream2,cudaStreamNonBlocking, low_priority);

  cudaMemcpy(transfer_count_ptr, wb_queue_counter + head_ptr, sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaMemset(wb_queue_counter + head_ptr, 0, sizeof(uint32_t));
  if(transfer_count_ptr[0] >= wb_queue_depth){
    transfer_count_ptr[0] = wb_queue_depth;
  }

  

  // printf("transfer_count: %llu\n", (unsigned long long) transfer_count_ptr[0]);
  // printf("memcpy counter: %llu\n", (unsigned long long) memcpy_counter);

  uint64_t* meta_array;
  uint64_t* loc_array;
  uint32_t memcpy_counter;
  //cpu_agg_buffer
  if(cpu_agg_flag){
    //std::cout << "cpu aggregate start\n";
    // cudaMemcpy(memcpy_count_ptr, (h_pc -> cpu_agg_queue_counter) + head_ptr, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    // uint32_t memcpy_counter = memcpy_count_ptr[0];
    // cudaMemset((h_pc -> cpu_agg_queue_counter) + head_ptr, 0, sizeof(uint32_t));
    
    //Async
    cudaMemcpy(memcpy_count_ptr, (h_pc -> cpu_agg_queue_counter) + head_ptr, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    memcpy_counter = memcpy_count_ptr[0];
    cudaMemset((h_pc -> cpu_agg_queue_counter) + head_ptr, 0, sizeof(uint32_t));
    


    meta_array = (h_pc -> h_cpu_agg_meta_queue) + cpu_agg_queue_depth * head_ptr;
    loc_array = (h_pc -> h_cpu_agg_loc_queue) + cpu_agg_queue_depth * head_ptr;
    auto t1 = Clock::now();
    //cpu_aggregate_kernel<float>(meta_array, memcpy_counter, host_wb_queue_ptr, wb_queue_depth, cpu_agg_buffer, head_ptr, wb_depth, dim );

    auto t2 = Clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1); // Microsecond (as int)
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1); // Microsecond (as int)
    auto ms_fractional =  static_cast<float>(us.count()) / 1000; // Milliseconds (as float)
   // std::cout << "cpu aggregat kernelTime: " << ms_fractional << std::endl;

    
  // printf("cpu aggregat kernel done\n");
  }
	

  
 
	TYPE* d_feature_ptr = (TYPE *) i_feature_ptr;
 	uint64_t* d_node_id_ptr = (uint64_t *) i_node_id_ptr;


  TYPE* wb_queue_head = host_wb_queue_ptr + pageSize / sizeof(TYPE) * wb_queue_depth * head_ptr;
  uint64_t* wb_id_array_head = h_wb_id_array + wb_queue_depth * head_ptr;

	cudaMemcpyAsync(d_feature_ptr, wb_queue_head, transfer_count_ptr[0] * pageSize, cudaMemcpyHostToDevice, transfer_stream);	
  if(cpu_agg_flag){
    cudaMemcpyAsync(d_agg_buffer, cpu_agg_buffer, pageSize * memcpy_counter, cudaMemcpyHostToDevice, transfer_stream2);
    cudaMemcpyAsync(d_agg_loc, loc_array, sizeof(uint64_t) * memcpy_counter, cudaMemcpyHostToDevice, transfer_stream2);
  }
  cudaMemcpyAsync(d_node_id_ptr, wb_id_array_head, transfer_count_ptr[0] * sizeof(uint64_t), cudaMemcpyHostToDevice, transfer_stream);	

  head_ptr = (head_ptr + 1) % wb_depth;

}


template <typename T = float>
__global__ 
void fill_batch_kernel(uint64_t* node_id_ptr, uint32_t* node_flag_ptr, int batch_size, int dim) {


  uint64_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < batch_size){
  uint64_t node_id = node_id_ptr[id];
//    printf("bath size: %lu node_id:%lu node idx: %lu\n",(unsigned long) batch_size, (unsigned long) node_id, (unsigned long) batch_src_node_id);
    
  	node_flag_ptr[node_id] = (uint32_t) id;
  }
}

template <typename T = float>
__global__ 
void fill_batch_kernel2(uint64_t* node_id_ptr, uint32_t* node_flag_ptr, int batch_size, int dim) {


  uint64_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < batch_size){
    uint64_t node_id = node_id_ptr[id];
    // if(id == 1)
    //   printf("bath size: %lu node_id:%lu node idx: %lu\n",(unsigned long) batch_size, (unsigned long) node_id, (unsigned long) id);
    
    // if(node_flag_ptr[node_id] != 0){
    //   atomicAdd(test_array, (unsigned int) 1);
    // }
  	node_flag_ptr[node_id] = (uint32_t) id;
  }
}


__global__
void print_node(uint64_t* node_id_ptr, uint32_t count){

  if(threadIdx.x == 0){
    for(uint32_t i =0; i < count; i++){
      printf("node id:%lu\n", (unsigned long) (node_id_ptr[i]));
    }
  }
}

void BAM_Feature_Store::fill_batch(uint64_t i_feature_ptr, uint64_t i_node_id_ptr, uint64_t i_batch_ptr, uint64_t i_batch_node_ptr, uint64_t node_flag_ptr, int batch_size, int dim, bool first){

	cudaStreamCreateWithFlags(&fill_stream, cudaStreamNonBlocking);
	if(first){
		printf("first\n");
		return;
	}
  //printf("fill batch start\n");

	TYPE* d_feature_ptr = (TYPE *) i_feature_ptr;
 	uint64_t* d_node_id_ptr = (uint64_t *) i_node_id_ptr;

  
  TYPE* d_batch_ptr = (TYPE *) i_batch_ptr;
  uint64_t* d_batch_node_ptr = (uint64_t*) i_batch_node_ptr;

  uint32_t* d_node_flag_ptr = (uint32_t*) node_flag_ptr;
  uint32_t count  = transfer_count_ptr[0];
  uint32_t memcpy_counter = memcpy_count_ptr[0];

  //print_node<<<1,32>>>(d_node_id_ptr, count);
 // printf("malloc start\n");
  
  cuda_err_chk(cudaDeviceSynchronize());
  
  
  //cuda_err_chk(cudaStreamSynchronize(transfer_stream));

  auto t1 =  Clock::now();
 
  prefetch_count += count;
  uint32_t g_size = (count + 127)/128; 
  fill_batch_kernel<TYPE><<<g_size, 128, 0, fill_stream>>>(d_node_id_ptr, d_node_flag_ptr, count, dim);
//  cuda_err_chk(cudaDeviceSynchronize());

  if(cpu_agg_flag){
    // unsigned int* test_array;
    // cudaMalloc(&test_array, sizeof(unsigned int));
    // cudaMemset(test_array, 0 ,sizeof(unsigned int) );
   // printf("counter: %llu\n", (unsigned long long) count);
    //printf("memcpy_counter: %llu\n", (unsigned long long) memcpy_counter);
    uint32_t g_size2 = (memcpy_counter + 127)/128; 
    memcpy_count += memcpy_counter;
    fill_batch_kernel2<TYPE><<<g_size2, 128, 0, fill_stream>>>(d_agg_loc, d_node_flag_ptr, memcpy_counter, dim);
  
    // //cuda_err_chk(cudaDeviceSynchronize());
    // unsigned int temp;
    // cudaMemcpy(&temp, test_array, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    // overlap += temp;
    // printf("temp: %lu\n", (unsigned long) temp);
    //   cudaFree(test_array);
  }

  cuda_err_chk(cudaDeviceSynchronize());
 //cudaStreamSynchronize(transfer_stream);
  auto t2 = Clock::now();
 
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(
      t2 - t1); // Microsecond (as int)
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      t2 - t1); // Microsecond (as int)
  const float ms_fractional =
      static_cast<float>(us.count()) / 1000; // Milliseconds (as float)

  
  cudaStreamDestroy(transfer_stream);
    cudaStreamDestroy(transfer_stream2);

  cudaStreamDestroy(wb_stream);
  fill_batch_time += ms_fractional;

  return;
}






void BAM_Feature_Store::read_feature(uint64_t i_ptr, uint64_t i_index_ptr,
                                     int64_t num_index, int dim, int cache_dim=1024) {




  TYPE *tensor_ptr = (TYPE *)i_ptr;
  int64_t *index_ptr = (int64_t *)i_index_ptr;

  uint64_t b_size = blkSize;
//  b_size=32;
  uint64_t n_warp = b_size / 32;
  //  uint64_t g_size = (num_index + n_warp - 1) / n_warp;
  uint64_t g_size = (num_index+n_warp - 1) / n_warp;

  cuda_err_chk(cudaDeviceSynchronize());
  auto t1 = Clock::now();
  read_feature_kernel<TYPE><<<g_size, b_size>>>(a->d_array_ptr, tensor_ptr,
                                                 index_ptr, dim, num_index, cache_dim);

  cuda_err_chk(cudaDeviceSynchronize());

  auto t2 = Clock::now();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(
      t2 - t1); // Microsecond (as int)
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      t2 - t1); // Microsecond (as int)
  const float ms_fractional =
      static_cast<float>(us.count()) / 1000; // Milliseconds (as float)

  //std::cout << "Duration = " << us.count() << "µs (" << ms_fractional << "ms)"
    //        << std::endl;
 
  kernel_time += ms_fractional;
  total_access += num_index;
  return;
}

void BAM_Feature_Store::read_feature_with_wb (uint64_t i_ptr, uint64_t i_index_ptr, uint64_t i_node_flag_ptr, uint64_t i_node_ptr,
                                              int64_t num_index, int dim, int cache_dim=1024) {

  //printf("read_feature_with_wb: %lu\n", (unsigned long) time_step);

  TYPE *tensor_ptr = (TYPE *)i_ptr;
  int64_t *index_ptr = (int64_t *)i_index_ptr;
  uint32_t* node_flag_ptr = (uint32_t*) i_node_flag_ptr;
  TYPE *node_ptr = (TYPE *) i_node_ptr;

  uint64_t b_size = blkSize;
  //b_size=32;
  uint64_t n_warp = b_size / 32;
  uint64_t g_size = (num_index+n_warp - 1) / n_warp;

  unsigned int * hit_counter;
  //cudaMalloc(&hit_counter, sizeof(unsigned int));
  //cudaMemset(hit_counter, 0, sizeof(unsigned int));

  auto t1 = Clock::now();

  total_access += num_index;

   cuda_err_chk(cudaDeviceSynchronize());
  read_feature_kernel_with_wb<TYPE><<<g_size, b_size, 0, fill_stream>>>(a->d_array_ptr, tensor_ptr, index_ptr, dim, num_index, cache_dim, 
                                                      wb_queue_counter,  wb_depth, wb_queue_ptr, wb_id_array, wb_queue_depth,
                                                      node_flag_ptr,
						      node_ptr,
                                                      time_step, head_ptr, 
                                                      hit_counter);
  cuda_err_chk(cudaDeviceSynchronize());

  /*
  unsigned int h_hit_counter;
  cudaMemcpy(&h_hit_counter, hit_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  printf("hit counter:%lu\n", (unsigned long) h_hit_counter);
  cudaFree(hit_counter);

  cudaMemset(hit_counter, 0, sizeof(bool) * transfer_count_ptr[0]);
*/

  auto t2 = Clock::now();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(
      t2 - t1); // Microsecond (as int)
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      t2 - t1); // Microsecond (as int)
  const float ms_fractional =
      static_cast<float>(us.count()) / 1000; // Milliseconds (as float)
 
  kernel_time += ms_fractional;
  
  return;
}

void BAM_Feature_Store::print_wb_queue(){
  for(int i = 0; i < wb_depth; i++){
    printf("Queue IDX: %i\n",i);
    for(int j = 0; j < wb_queue_depth; j++){
      //std::cout << host_wb_queue_ptr[i * wb_queue_depth + j] << std::endl;
      printf("%lu\n",(unsigned long) (h_wb_id_array[i * wb_queue_depth + j]));

    }
  }
}

void BAM_Feature_Store::update_time(){
  time_step++;
  if(time_step == 255) time_step = 0;
}


void  BAM_Feature_Store::init_cpu_meta(uint64_t num_cl){
  printf("num cl:%llu\\n", (unsigned long long)num_cl);
  cuda_err_chk(cudaHostAlloc(&host_meta, num_cl * sizeof(uint64_t), cudaHostAllocMapped));
  cudaHostGetDevicePointer((uint64_t **)&device_meta, (uint64_t *)host_meta, 0);

}

template <typename T = float>
__global__
void set_wb_counter_cpu_kernel(array_d_t<T> *dr, uint64_t* meta_ptr, uint32_t num_cl){

    uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

     uint64_t update_reuse = meta_ptr[tid];
      if(update_reuse == 1) printf("node 1\n");
      /*
    if(tid < num_cl){
        wb_bam_ptr<T> ptr(dr);
	uint64_t node_id = ptr.get_page_id(tid);
    	
	uint64_t update_reuse = meta_ptr[node_id];
	if(update_reuse == 1) printf("node 1\n");
    }
    */

}

void  BAM_Feature_Store::set_wb_counter_with_CPU(uint64_t batch_array_idx, uint64_t batch_size_idx, uint32_t max_batch_size){

	uint32_t num_g = (numPages + 127) / 128;
	printf("num pages: %llu\n", (unsigned long long) numPages);
	 cuda_err_chk(cudaDeviceSynchronize())
	auto t1 = Clock::now();
	set_wb_counter_cpu_kernel<TYPE><<<num_g,128>>>(a->d_array_ptr, device_meta, numPages);
        cuda_err_chk(cudaDeviceSynchronize())
	
	auto t2 = Clock::now();
 	auto us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1); // Microsecond (as int)
  	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1); // Microsecond (as int)
  	auto ms_fractional =  static_cast<float>(us.count()) / 1000; // Milliseconds (as float)

	std::cout << "set up counter cpu Kernel Time: " << ms_fractional << std::endl;
}



template<typename T >
void  cpu_aggregate_kernel_test(uint64_t* list_of_src, uint64_t list_len,  T * pinned_queue_ptr, uint64_t q_size,  T * dst_buf, uint64_t dim){

  #define encode_mask 0x00000000FFFFFFFF
  
  for(uint64_t i = 0; i < list_len; i++){
    uint64_t encoded_src = list_of_src[i];
    uint64_t q_idx = (encoded_src & encode_mask);

    uint64_t head_idx = (encoded_src >> 32);
    //std::cout << "head : " << head_idx << " idx: " << q_idx << std::endl;
    T* src_ptr = pinned_queue_ptr + head_idx * q_size + q_idx;
    T* dst_ptr = dst_buf + i * dim;
    std::memcpy(dst_ptr, src_ptr, dim * sizeof(uint64_t)); // OK
  }
};

void  BAM_Feature_Store::cpu_aggregate(uint64_t dim){
  #define encode_mask 0x00000000FFFFFFFF
  printf("cpu aggregat start dim:%llu\n", (unsigned long long) dim);

  uint64_t test_c = 10000;
  uint64_t* test_ptr = (uint64_t*) malloc(test_c * sizeof(uint64_t));
  for(uint64_t i = 0; i < test_c; i++){
    uint64_t head = (rand() % 128);
    uint64_t idx = (rand() % (64 * 1024));
    uint64_t encode_line = ((head << 32) | (idx));
    test_ptr[i] = encode_line;

  }
  printf("cpu aggregat setup done\n");
	auto t1 = Clock::now();

  cpu_aggregate_kernel_test<float>(test_ptr, test_c , host_wb_queue_ptr, wb_queue_depth, cpu_agg_ptr, dim);
  
	auto t2 = Clock::now();
 	auto us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1); // Microsecond (as int)
  	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1); // Microsecond (as int)
  	auto ms_fractional =  static_cast<float>(us.count()) / 1000; // Milliseconds (as float)

	std::cout << "cpu aggregat kernelTime: " << ms_fractional << std::endl;
  printf("cpu aggregat kernel done\n");
  free(test_ptr);

}




PYBIND11_MODULE(BAM_Feature_Store, m) {
  m.doc() = "Python bindings for an example library";

  namespace py = pybind11;

  py::class_<BAM_Feature_Store,
             std::unique_ptr<BAM_Feature_Store, py::nodelete>>(
      m, "BAM_Feature_Store")
      .def(py::init([]() { return new BAM_Feature_Store(); }))
      .def("init_controllers", &BAM_Feature_Store::init_controllers)
      .def("mgc_init_controllers", &BAM_Feature_Store::mgc_init_controllers)
      .def("print_stats", &BAM_Feature_Store::print_stats)
      .def("read_feature", &BAM_Feature_Store::read_feature)
      .def("read_feature_with_wb", &BAM_Feature_Store::read_feature_with_wb)
      .def("pin_pages", &BAM_Feature_Store::pin_memory)
      .def("set_prefetching", &BAM_Feature_Store::set_prefetching)
      .def("set_window_buffering", &BAM_Feature_Store::set_window_buffering)
      .def("init_backing_memory", &BAM_Feature_Store::init_backing_memory)
      .def("fetch_from_backing_memory", &BAM_Feature_Store::fetch_from_backing_memory)
      .def("fetch_from_backing_memory_chunk", &BAM_Feature_Store::fetch_from_backing_memory_chunk)
      .def("create_streams", &BAM_Feature_Store::create_streams)
      .def("sync_streams", &BAM_Feature_Store::sync_streams)
      .def("print_wb_queue", &BAM_Feature_Store::print_wb_queue)
      .def("set_wb_counter", &BAM_Feature_Store::set_wb_counter)
      .def("set_wb_counter_list", &BAM_Feature_Store::set_wb_counter_list)
      .def("prefetch_from_victim_queue", &BAM_Feature_Store::prefetch_from_victim_queue)
      .def("fill_batch", &BAM_Feature_Store::fill_batch)
      .def("set_wb_counter_with_CPU", &BAM_Feature_Store::set_wb_counter_with_CPU)
      .def("init_cpu_meta", &BAM_Feature_Store::init_cpu_meta)
      .def("cpu_aggregate", &BAM_Feature_Store::cpu_aggregate)
      .def("update_time", &BAM_Feature_Store::update_time);
}


