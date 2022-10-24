#include <pybind11/pybind11.h>

#include <string>
#include <stdexcept>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <fstream>

#include <stdio.h>
#include <cuda.h>
#include <nvm_ctrl.h>
#include <nvm_types.h>
#include <nvm_queue.h>
#include <nvm_util.h>
#include <nvm_admin.h>
#include <nvm_error.h>
#include <nvm_cmd.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <buffer.h>
#include <nvm_parallel_queue.h>
#include <nvm_io.h>
#include <util.h>

#include <ctrl.h>
#include <page_cache.h>
#include <event.h>
#include <queue.h>
#include <stdio.h>
#include <vector>

#include <bam_nvme.h>

//void BAM_Feature_Store::init_controllers(const char *const ctrls_paths[]) {
void BAM_Feature_Store::init_controllers() {

 const char* const ctrls_paths[] = {"/dev/libnvmpro0"};

  // set device might requrie here
  // cuda_err_chk(cudaSetDevice(settings.cudaDevice));
        std::vector<Controller *> ctrls(n_ctrls);
  for (size_t i = 0; i < n_ctrls; i++) {
    ctrls[i] = new Controller(ctrls_paths[i], nvmNamespace, cudaDevice,
                              queueDepth, numQueues);
  }

  uint64_t b_size = blkSize;
  uint64_t g_size = (numThreads + b_size - 1) / b_size;
  uint64_t n_threads = b_size * g_size;

  uint64_t page_size = pageSize;
  uint64_t n_pages = numPages;
  uint64_t total_cache_size = (page_size * n_pages);

 page_cache_t h_pc(page_size, n_pages, cudaDevice, ctrls[0][0], (uint64_t)64,
                    ctrls);
  page_cache_t *d_pc = (page_cache_t *)(h_pc.d_pc_ptr);
#define TYPE uint64_t
  uint64_t t_size = numElems * sizeof(TYPE);

  range_t<uint64_t> h_range((uint64_t)0, (uint64_t)numElems, (uint64_t)0,
                            (uint64_t)(t_size / page_size), (uint64_t)0,
                            (uint64_t)page_size, &h_pc, cudaDevice);
  range_t<uint64_t> *d_range = (range_t<uint64_t> *)h_range.d_range_ptr;


  std::vector<range_t<uint64_t>*> vr(1);
  vr[0] = & h_range;
 // array_t<uint64_t> a(n_elems, 0, vr, settings.cudaDevice);
  return;
}


PYBIND11_MODULE(BAM_Feature_Store, m)
{
  m.doc() = "Python bindings for an example library";

  namespace py = pybind11;

  py::class_<BAM_Feature_Store>(m, "Example")
    .def( py::init( []()
            {
              return new BAM_Feature_Store();
            }
          )
    )
    .def("init_controllers", &BAM_Feature_Store::init_controllers);
    ;
    

}
