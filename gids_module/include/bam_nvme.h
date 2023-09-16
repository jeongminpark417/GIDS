#ifndef BAMNVME_H
#define BAMNVME_H

#define TYPE float

struct BAM_Feature_Store {
  const char *const ctrls_paths[5] = {"/dev/libnvm1","/dev/libnvm0","/dev/libnvm2","/dev/libnvm3","/dev/libnvm4"};

  cudaStream_t stream_array[8];
  cudaStream_t transfer_stream;
  cudaStream_t transfer_stream2;
  cudaStream_t wb_stream;
  cudaStream_t fill_stream;
  int dim;
  bool cpu_agg_flag;
  uint64_t total_access;
  uint64_t prefetch_count;
  uint64_t memcpy_count;
  uint64_t overlap;

  uint32_t cpu_agg_queue_depth;
  TYPE* cpu_agg_buffer;
  TYPE* d_agg_buffer;
  uint64_t* d_agg_loc; 

  uint32_t** d_batch_array_ptr;


  float* h_buf_ptr;
  float* d_buf_ptr;

  uint32_t cudaDevice = 0;
  uint64_t cudaDeviceId = 0;
//  const char *blockDevicePath = nullptr;
//  const char *controllerPath = nullptr;
  uint64_t controllerId = 0;
  uint32_t adapter = 0;
  uint32_t segmentId = 0;
  uint32_t nvmNamespace = 1;
  bool doubleBuffered = false;
  size_t numReqs = 100;
  size_t numPages = 262144 * 8   ;
  // size_t numPages = 1024*100 ;
  
  size_t startBlock = 0;


  bool stats = false;
//  const char *output = nullptr;
  size_t numThreads = 64;
  uint32_t domain = 0;
  uint32_t bus = 0;
  uint32_t devfn = 0;

  uint32_t n_ctrls = 1;
  size_t blkSize = 128;
  size_t queueDepth = 1024;
  size_t numQueues = 128;
  size_t pageSize = 4096 ;
  uint64_t numElems = 300LL*1000*1000*1024;

  uint64_t read_offset = 0;
  std::vector<Controller *> ctrls;
  page_cache_t *h_pc;
  range_t<TYPE> *h_range;

  //wb
  uint32_t* transfer_count_ptr;
  uint32_t* memcpy_count_ptr;

  uint32_t wb_depth = 4;
  uint32_t wb_queue_depth =  128 * 1024;
  //uint32_t wb_queue_depth = 64;
  
  TYPE* wb_queue_ptr;
  TYPE* host_wb_queue_ptr;
  TYPE* cpu_agg_ptr;
  uint32_t* wb_queue_counter;

  uint64_t* h_wb_id_array;
  uint64_t* wb_id_array;
        
  uint8_t time_step;
  int32_t head_ptr;
  
  std::vector<range_t<TYPE> *> vr;
  array_t<TYPE> *a;
  range_d_t<TYPE> *d_range;

  BAM_Feature_Store()  {
  };

  ~BAM_Feature_Store(){
 
	  for(auto i : ctrls){
	  	delete(i);
	  }
	  delete(h_pc);
	  delete(h_range);
	  delete(a);
  }
  // BAM_Feature_Store(const std::string &name)
  //   : name(name) {}
  // void init_controllers(const char* const ctrls_paths[], uint32_t
  // nvmNamespace, uint32_t cudaDevice, uint64_t queueDepth, uint64_t numQueues,
  // int num_controllers, std::vector<Controller*> &ctrls_vec);

 uint16_t reset_counter = 0;
uint64_t* host_meta;
uint64_t* device_meta;


  float kernel_time = 0; 
  float fill_batch_time = 0;
  float set_wb_time = 0;
  float flush_time = 0; 

  int low_priority;
  int high_priority;

  void print();
  int add(int a, int b);

  void init_controllers(int ps, uint64_t r_off, uint64_t num_ele, uint64_t cache_size,uint64_t num_ssd, uint32_t wb_size, uint64_t q_size, bool f, int32_t cpu_agg_q_depth);
  void mgc_init_controllers(int ps, uint64_t r_off, uint64_t num_ele, uint64_t cache_size,uint64_t ctrl_idx, bool cpu_cache, uint64_t cpu_cache_ptr);

  void read_feature_test();
  void read_feature(uint64_t tensor_ptr, uint64_t index_ptr,int64_t num_index, int dim, int cache_dim);
  void print_stats();
  void pin_memory(uint64_t i_index_ptr, int64_t num_pin_page, int dim);	
  void set_prefetching(uint64_t id_idx, uint64_t prefetch_idx, int64_t num_pages);
  void set_window_buffering(uint64_t id_idx, uint64_t prefetch_idx, int64_t num_pages); 


  //multi-GPU
  void init_backing_memory(size_t memory_size);
  void fetch_from_backing_memory(uint64_t i_device_ptr, uint64_t i_batch_idx_ptr, uint64_t i_backing_idx_ptr,  int batch_size, int cl_size, int num_tranfer_cl);
  void fetch_from_backing_memory_chunk(uint64_t i_device_ptr, uint64_t cl_size, int stream_id);
  void create_streams(int num_streams);
  void sync_streams(int num_streams);

  void set_wb_counter(uint64_t batch_array_idx, uint64_t batch_size_idx, uint32_t max_size);
  void read_feature_with_wb (uint64_t i_ptr, uint64_t i_index_ptr, uint64_t i_node_flag_ptr, uint64_t i_node_ptr, int64_t num_index, int dim, int cache_dim);
  void prefetch_from_victim_queue(uint64_t i_feature_ptr, uint64_t i_node_id_ptr, int stream_id);
  void fill_batch(uint64_t i_feature_ptr, uint64_t i_node_id_ptr, uint64_t i_batch_ptr, uint64_t i_batch_node_ptr, uint64_t node_flag_ptr, int batch_size, int dim, bool first);


void set_wb_counter_list(uint64_t batch_array_idx, uint64_t batch_size_idx, uint32_t max_batch_size);
void set_wb_counter_with_CPU(uint64_t batch_array_idx, uint64_t batch_size_idx, uint32_t max_size);
void init_cpu_meta(uint64_t num_cl);


void cpu_aggregate(uint64_t dim);

  void print_wb_queue();
  void update_time();
//  void fetch_from_backing_memory();
};

#endif
