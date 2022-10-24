struct BAM_Feature_Store {
  uint32_t cudaDevice = 0;
  uint64_t cudaDeviceId = 0;
  const char *blockDevicePath = nullptr;
  const char *controllerPath = nullptr;
  uint64_t controllerId = 0;
  uint32_t adapter = 0;
  uint32_t segmentId = 0;
  uint32_t nvmNamespace = 1;
  bool doubleBuffered = false;
  size_t numReqs = 1;
  size_t numPages = 1024;
  size_t startBlock = 0;
  bool stats = false;
  const char *output = nullptr;
  size_t numThreads = 64;
  uint32_t domain = 0;
  uint32_t bus = 0;
  uint32_t devfn = 0;

  uint32_t n_ctrls = 1;
  size_t blkSize = 64;
  size_t queueDepth = 16;
  size_t numQueues = 1;
  size_t pageSize = 4096;
  uint64_t numElems = 1024;
 // std::string name;
  
  BAM_Feature_Store(){}
  //BAM_Feature_Store(const std::string &name)
   //   : name(name) {}
  // void init_controllers(const char* const ctrls_paths[], uint32_t
  // nvmNamespace, uint32_t cudaDevice, uint64_t queueDepth, uint64_t numQueues,
  // int num_controllers, std::vector<Controller*> &ctrls_vec);

  void print();
  int add (int a, int b);
  void init_controllers();
};
