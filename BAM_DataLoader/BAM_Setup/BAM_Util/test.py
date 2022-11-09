import BAM_Util
from BAM_Util import BAM_Util

import BAM_Feature_Store
import torch

test = BAM_Util(100,100)

c_index = torch.tensor([0,1,2], dtype=torch.long)
index = c_index.to('cuda:0')
test_gpu = test.fetch_feature(index, 100)

print(test_gpu)
