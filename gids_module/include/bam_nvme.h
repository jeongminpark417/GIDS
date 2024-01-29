#ifndef BAMNVME_H
#define BAMNVME_H

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

//#define TYPE float
struct GIDS_Controllers {
  const char *const ctrls_paths[6] = {"/dev/libnvm0","/dev/libnvm1","/dev/libnvm2","/dev/libnvm3","/dev/libnvm4","/dev/libnvm5"};
  std::vector<Controller *> ctrls;

  uint32_t n_ctrls = 1;
  uint64_t queueDepth = 1024;
  uint64_t numQueues = 128;
  
  uint32_t cudaDevice = 0;
  uint32_t nvmNamespace = 1;
  
  //member functions
  void init_GIDS_controllers(uint32_t num_ctrls, uint64_t q_depth, uint64_t num_q,  const std::vector<int>& ssd_list);

};

template <typename TYPE>
struct GIDS_CPU_buffer {
    TYPE* cpu_buffer;
    TYPE* device_cpu_buffer;
    uint64_t cpu_buffer_dim;
    uint64_t cpu_buffer_len;
};


template <typename TYPE>
struct BAM_Feature_Store {


  GIDS_CPU_buffer<TYPE> CPU_buffer;
  //GIDS optimization flasg
  bool cpu_buffer_flag = false;
  bool seq_flag = true;
  //Sampling Offsets
  uint64_t* offset_array;

  int dim;
  uint64_t total_access;
  unsigned int cpu_access_count = 0;
  unsigned int* d_cpu_access;

  //BAM parameters
  uint32_t cudaDevice = 0;
  size_t numPages = 262144 * 8;
  bool stats = false;
  size_t numThreads = 64;
  uint32_t domain = 0;
  uint32_t bus = 0;
  uint32_t devfn = 0;

  uint32_t n_ctrls = 1;
  size_t blkSize = 128;
  size_t queueDepth = 1024;
  size_t numQueues = 128;
  uint32_t pageSize = 4096 ;
  uint64_t numElems = 300LL*1000*1000*1024;
  uint64_t read_offset = 0;
  std::vector<Controller *> ctrls;

  page_cache_t *h_pc;
  range_t<TYPE> *h_range;
  std::vector<range_t<TYPE> *> vr;
  array_t<TYPE> *a;
  range_d_t<TYPE> *d_range;
  //wb

  
  float kernel_time = 0; 


  void init_controllers(GIDS_Controllers GIDS_ctrl, uint32_t ps, uint64_t r_off, uint64_t num_ele, uint64_t cache_size, 
                        uint64_t num_ssd);

  void read_feature(uint64_t tensor_ptr, uint64_t index_ptr,int64_t num_index, int dim, int cache_dim, uint64_t key_off);
  void read_feature_hetero(int num_iter, const std::vector<uint64_t>&  i_ptr_list, const std::vector<uint64_t>& i_index_ptr_list, const std::vector<uint64_t>&   num_index, int dim, int cache_dim, const std::vector<uint64_t>& key_off);
  void read_feature_merged(int num_iter, const std::vector<uint64_t>&  i_ptr_list, const std::vector<uint64_t>& i_index_ptr_list, const std::vector<uint64_t>&   num_index, int dim, int cache_dim);
  void read_feature_merged_hetero(int num_iter, const std::vector<uint64_t>&  i_ptr_list, const std::vector<uint64_t>& i_index_ptr_list, const std::vector<uint64_t>&   num_index, int dim, int cache_dim, const std::vector<uint64_t>& key_off);

  void cpu_backing_buffer(uint64_t dim, uint64_t len);
  void set_cpu_buffer(uint64_t idx_buffer, int num);  

  void set_window_buffering(uint64_t id_idx,  int64_t num_pages, int hash_off); 
  void print_stats();
  void print_stats_no_ctrl();

 
  uint64_t get_array_ptr();
  uint64_t get_offset_array();
  void set_offsets(uint64_t in_off, uint64_t index_off, uint64_t data_off);
  void store_tensor(uint64_t tensor_ptr, uint64_t num, uint64_t offset);
  void read_tensor( uint64_t num, uint64_t offset);
  void flush_cache();
  unsigned int get_cpu_access_count();
  void flush_cpu_access_count();

};

#endif
