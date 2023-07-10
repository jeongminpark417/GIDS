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
  uint64_t n_pages = numPages;
  n_pages = cache_size * 1024LL*1024*1024/page_size;
//  n_pages = 10;
  uint64_t total_cache_size = (page_size * n_pages);

  std::cout << "n pages: " << n_pages <<std::endl;
  std::cout << "page size: " << pageSize << std::endl;
  std::cout << "num elements: " << numElems << std::endl;

  h_pc = new page_cache_t(page_size, n_pages, cudaDevice, ctrls[0][0],
                          (uint64_t)64, ctrls);


  page_cache_t *d_pc = (page_cache_t *)(h_pc->d_pc_ptr);
  uint64_t t_size = numElems * sizeof(TYPE);

  std::cout << "numElems: " << numElems << std::endl;
  printf("numElems: %llu t_size:%llu page_size: %llu\n",(uint64_t)numElems,(uint64_t)(t_size),  (uint64_t)page_size);
  h_range = new range_t<TYPE>((uint64_t)0, (uint64_t)numElems, (uint64_t)read_off,
                              (uint64_t)(t_size / page_size), (uint64_t)0,
                              (uint64_t)page_size, h_pc, cudaDevice, 
			      //REPLICATE
			      STRIPE
			      );

  range_t<TYPE> *d_range = (range_t<TYPE> *)h_range->d_range_ptr;

  // std::vector<range_t<TYPE>*> vr(1);
  vr.push_back(nullptr);
  vr[0] = h_range;

  //  size_t cpu_cache_size = cpu_page_num * page_size;
   a = new array_t<TYPE>(numElems, 0, vr, cudaDevice);


  printf("init done\n");

  return;
}

// void BAM_Feature_Store::init_controllers(const char *const ctrls_paths[]) {
void BAM_Feature_Store::init_controllers(int ps, uint64_t read_off, uint64_t cache_size, uint64_t num_ele = 100, uint64_t num_ssd = 1, bool cpu_cache = false, uint64_t cpu_page_num = 0) {

  printf("init starts\n");
  numElems = num_ele;
  read_offset = read_off;
  n_ctrls = num_ssd;

  for (size_t i = 0; i < num_ssd; i++) {
    ctrls.push_back(new Controller(ctrls_paths[i], nvmNamespace, cudaDevice,
                                   queueDepth, numQueues));
  }
  printf("controllers are initalized\n");
  uint64_t b_size = blkSize;
  uint64_t g_size = (numThreads + b_size - 1) / b_size;
  // uint64_t g_size = (numThreads) / b_size;
  uint64_t n_threads = b_size * g_size;

  pageSize = ps;
  uint64_t page_size = pageSize;
  uint64_t n_pages = numPages;
  n_pages = cache_size * 1024LL*1024*1024/page_size;
//  n_pages = 10;
  uint64_t total_cache_size = (page_size * n_pages);

  std::cout << "n pages: " << n_pages <<std::endl;
  std::cout << "page size: " << pageSize << std::endl;
  std::cout << "num elements: " << numElems << std::endl;

  h_pc = new page_cache_t(page_size, n_pages, cudaDevice, ctrls[0][0],
                          (uint64_t)64, ctrls);


  page_cache_t *d_pc = (page_cache_t *)(h_pc->d_pc_ptr);
  uint64_t t_size = numElems * sizeof(TYPE);

  // error
  /*
  range_t<TYPE> h_range2((uint64_t)0, (uint64_t)numElems, (uint64_t)0,
                        (uint64_t)(t_size / page_size), (uint64_t)0,
                        (uint64_t)page_size, h_pc, cudaDevice);
*/
  printf("create h_range\n");
  std::cout << "numElems: " << numElems << std::endl;
  printf("numElems: %llu t_size:%llu page_size: %llu\n",(uint64_t)numElems,(uint64_t)(t_size),  (uint64_t)page_size);
  h_range = new range_t<TYPE>((uint64_t)0, (uint64_t)numElems, (uint64_t)read_off,
                              (uint64_t)(t_size / page_size), (uint64_t)0,
                              (uint64_t)page_size, h_pc, cudaDevice, 
			      REPLICATE
			      //STRIPE
			      );

  // h_range = std::move(h_range2);

  printf("h rnage created\n");
  range_t<TYPE> *d_range = (range_t<TYPE> *)h_range->d_range_ptr;

  // std::vector<range_t<TYPE>*> vr(1);
  vr.push_back(nullptr);
  vr[0] = h_range;

  /*
  array_t<TYPE> a2(numElems, 0, vr, cudaDevice);
  a = std::move(a2);
*/
  printf("array_t crate\n");
//  size_t cpu_cache_size = cpu_page_num * page_size;
   a = new array_t<TYPE>(numElems, 0, vr, cudaDevice);


  printf("init done\n");

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
	/*
	uint64_t c_size = 1024ULL * 1024ULL;
	uint64_t num_p = cl_size/(c_size);
	for(uint64_t i =0; i< num_p; i++){
		cudaMemcpyAsync(device_ptr +c_size*i/4, h_buf_ptr + c_size*i/4, c_size, cudaMemcpyHostToDevice, stream_array[stream_id]);
	}
	*/
}



__global__  void compte_test_kernel(int* ptr, int num_idx){

	int out = 0;
	if(threadIdx.x == 0){
		for(int i  = 0; i < num_idx-1; i++){
			int temp = ptr[i];
			temp += (ptr[i+1] >> 1);
			out += temp;
		}
	ptr[0] = out;
	}
}


void BAM_Feature_Store::compute_test(uint64_t i_ptr, int num_idx){
	

	int* d_ptr = (int*) i_ptr;	
	compte_test_kernel<<<1,32, 0, stream_array[7]>>>(d_ptr, num_idx);
//	cudaDeviceSynchronize();
}

void BAM_Feature_Store::read_feature(uint64_t i_ptr, uint64_t i_index_ptr,
                                     int64_t num_index, int dim, int cache_dim=1024) {

  typedef std::chrono::high_resolution_clock Clock;



  TYPE *tensor_ptr = (TYPE *)i_ptr;
  int64_t *index_ptr = (int64_t *)i_index_ptr;

  uint64_t b_size = blkSize;
  b_size=32;
  uint64_t n_warp = b_size / 32;
  //  uint64_t g_size = (num_index + n_warp - 1) / n_warp;
  uint64_t g_size = (num_index+n_warp - 1) / n_warp;
 // g_size = 1;
  //`printf("g size: %llu\n", g_size);

  //printf("dim: %i\n", dim);
 // printf("num idx: %lli\n", num_index);

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
  return;
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
      .def("pin_pages", &BAM_Feature_Store::pin_memory)
      .def("set_prefetching", &BAM_Feature_Store::set_prefetching)
      .def("set_window_buffering", &BAM_Feature_Store::set_window_buffering)
      .def("init_backing_memory", &BAM_Feature_Store::init_backing_memory)
      .def("fetch_from_backing_memory", &BAM_Feature_Store::fetch_from_backing_memory)
      .def("fetch_from_backing_memory_chunk", &BAM_Feature_Store::fetch_from_backing_memory_chunk)
      .def("create_streams", &BAM_Feature_Store::create_streams)
      .def("sync_streams", &BAM_Feature_Store::sync_streams)
      .def("compute_test", &BAM_Feature_Store::compute_test);
}


