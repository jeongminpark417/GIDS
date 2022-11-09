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

template <typename T = float>
__global__ void  read_feature_kernel (array_d_t<T> *dr, 
					 float* out_tensor_ptr,
					 int64_t* index_ptr,
					 int dim,
					 int64_t num_idx) {
  uint64_t bid = blockIdx.x;
  int num_warps = blockDim.x / 32;
  int warp_id = blockIdx.x / 32;
  int idx_idx = bid * num_warps + warp_id;
  if(idx_idx < num_idx){
  int64_t row_index = index_ptr[idx_idx];
  

 // int64_t row_index = bid * num_warps + warp_id;


  uint64_t tid = threadIdx.x % 32;

  for(; tid < dim; tid += 32){
  	out_tensor_ptr[(bid* num_warps + warp_id)*dim + tid] = (*dr)[row_index * dim + tid];
  }
  }
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

// void BAM_Feature_Store::init_controllers(const char *const ctrls_paths[]) {
void BAM_Feature_Store::init_controllers() {

  printf("init starts\n");

  // set device might requrie here
  // cuda_err_chk(cudaSetDevice(settings.cudaDevice));
  // std::vector<Controller *> ctrls(n_ctrls);

  for (size_t i = 0; i < n_ctrls; i++) {
    //    ctrls[i] = new Controller(ctrls_paths[i], nvmNamespace, cudaDevice,
    //      queueDepth, numQueues);
    ctrls.push_back(new Controller(ctrls_paths[i], nvmNamespace, cudaDevice,
                                   queueDepth, numQueues));
  }

  uint64_t b_size = blkSize;
  uint64_t g_size = (numThreads + b_size - 1) / b_size;
 // uint64_t g_size = (numThreads) / b_size;
  uint64_t n_threads = b_size * g_size;

  uint64_t page_size = pageSize;
  uint64_t n_pages = numPages;
  uint64_t total_cache_size = (page_size * n_pages);

//  page_cache_t h_pc2(page_size, n_pages, cudaDevice, ctrls[0][0], (uint64_t)64,
 //                   ctrls);

  h_pc = new page_cache_t(page_size, n_pages, cudaDevice, ctrls[0][0], (uint64_t)64,
                    ctrls);
  
  // h_pc = std::move(h_pc2);
  
  page_cache_t * d_pc = (page_cache_t *)(h_pc->d_pc_ptr);
  uint64_t t_size = numElems * sizeof(TYPE);

  //error
  /*
  range_t<TYPE> h_range2((uint64_t)0, (uint64_t)numElems, (uint64_t)0,
                        (uint64_t)(t_size / page_size), (uint64_t)0,
                        (uint64_t)page_size, h_pc, cudaDevice);
*/
  h_range = new range_t<TYPE>((uint64_t)0, (uint64_t)numElems, (uint64_t)0,
                        (uint64_t)(t_size / page_size), (uint64_t)0,
                        (uint64_t)page_size, h_pc, cudaDevice);

 // h_range = std::move(h_range2);


   range_t<TYPE> *d_range = (range_t<TYPE> *)h_range -> d_range_ptr;

  // std::vector<range_t<TYPE>*> vr(1);
  vr.push_back(nullptr);
  vr[0] = h_range;
	
  /*
  array_t<TYPE> a2(numElems, 0, vr, cudaDevice);
  a = std::move(a2);
*/
  a = new array_t<TYPE>(numElems, 0, vr, cudaDevice);


  printf("init done\n");
  /*
  unsigned long long *d_req_count;
  cuda_err_chk(cudaMalloc(&d_req_count, sizeof(unsigned long long)));
  cuda_err_chk(cudaMemset(d_req_count, 0, sizeof(unsigned long long)));

  char st[15];
  cuda_err_chk(cudaDeviceGetPCIBusId(st, 15, cudaDevice));
  std::cout << st << std::endl;

  sequential_access_kernel<float>
      <<<g_size, b_size>>>(a.d_array_ptr, n_threads, d_req_count, numReqs);

  cuda_err_chk(cudaDeviceSynchronize());



  cuda_err_chk(cudaFree(d_req_count));
*/
  return;
}

void BAM_Feature_Store::read_feature_test() {

	/*
  printf("num Req: %i\n", numReqs);
  uint64_t b_size = blkSize;
  uint64_t g_size = (numThreads + b_size - 1) / b_size;
  uint64_t n_threads = b_size * g_size;

  unsigned long long *d_req_count;
  cuda_err_chk(cudaMalloc(&d_req_count, sizeof(unsigned long long)));
  cuda_err_chk(cudaMemset(d_req_count, 0, sizeof(unsigned long long)));
  std::cout << "relaunch kernel\n";

  char st[15];
  cuda_err_chk(cudaDeviceGetPCIBusId(st, 15, cudaDevice));
  std::cout << st << std::endl;


  sequential_access_kernel<float><<<g_size, b_size>>>(
      a.d_array_ptr, n_threads, d_req_count, numReqs);

  cuda_err_chk(cudaDeviceSynchronize());

  cuda_err_chk(cudaFree(d_req_count));
*/
  printf("read feature done\n");
  return;
}

void BAM_Feature_Store::read_feature(uint64_t i_ptr, uint64_t i_index_ptr, int64_t num_index, int dim) {

  typedef std::chrono::high_resolution_clock Clock;

  float* tensor_ptr = (float*)i_ptr;
  int64_t* index_ptr = (int64_t*) i_index_ptr;

  uint64_t b_size = blkSize;
  uint64_t n_warp = b_size/32;
//  uint64_t g_size = (num_index + n_warp - 1) / n_warp;
  uint64_t g_size = (num_index) / n_warp;

//  printf("g size: %llu\n", g_size);

  printf("dim: %i\n", dim);
  printf("num idx: %lli\n", num_index);

  auto t1 = Clock::now();
  read_feature_kernel<float><<<g_size, b_size>>>(
      a->d_array_ptr,  tensor_ptr, index_ptr, dim, num_index);

  cuda_err_chk(cudaDeviceSynchronize());

  auto t2 = Clock::now();
      auto us = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1); // Microsecond (as int)
   auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1); // Microsecond (as int)
    const float ms_fractional = static_cast<float>(us.count()) / 1000;         // Milliseconds (as float)

        std::cout << "Duration = " << us.count() << "µs (" << ms_fractional << "ms)" << std::endl;

	std::cout << "print rest: ";
	h_pc->print_reset_stats();
	std::cout << std::endl;

	std::cout << "print array rest: ";
	a ->print_reset_stats();
	std::cout << std::endl;
	
	std::cout << "print ctrl rest: ";
	(ctrls[0]) -> print_reset_stats();
	std::cout << std::endl;
	return;
}

PYBIND11_MODULE(BAM_Feature_Store, m) {
  m.doc() = "Python bindings for an example library";

  namespace py = pybind11;

  py::class_<BAM_Feature_Store, std::unique_ptr<BAM_Feature_Store, py::nodelete>>(m, "BAM_Feature_Store")
      .def(py::init([]() { return new BAM_Feature_Store(); }))
      .def("init_controllers", &BAM_Feature_Store::init_controllers)
      .def("read_feature_test", &BAM_Feature_Store::read_feature_test)
      .def("read_feature", &BAM_Feature_Store::read_feature);
}
