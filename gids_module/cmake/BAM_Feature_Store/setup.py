from distutils.core import setup

import sys
if sys.version_info < (3,0):
  sys.exit('Sorry, Python < 3.0 is not supported')

setup(
  name        = 'cmake_cpp_pybind11',
  version     = '0.1.1', # TODO: might want to use commit ID here
  packages    = [ 'BAM_Feature_Store' ],
  package_dir = {
    '': '/root/GIDS/gids_module/cmake'
  },
  package_data = {
    '': ['BAM_Feature_Store.so']
  }
)
