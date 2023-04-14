import BAM_Util

import torch

test = BAM_Util.BAM_Util()

c_index = torch.tensor([177644,16687,35082], dtype=torch.long)
index = c_index.to('cuda:0')
test_gpu = test.fetch_feature(index, 100)

print(test_gpu)
