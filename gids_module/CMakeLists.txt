CMAKE_MINIMUM_REQUIRED( VERSION 3.3.0 )
PROJECT( cmake-cpp-pybind11 CXX CUDA)

SET( CMAKE_CXX_STANDARD 11 )
SET( CMAKE_CXX_STANDARD_REQUIRED ON )

LIST( APPEND CMAKE_MODULE_PATH
  ${CMAKE_SOURCE_DIR}/cmake
  ${CMAKE_SOURCE_DIR}/cmake/Modules
)

########################################################################
# Macros and additional flags
########################################################################

INCLUDE( CheckCXXCompilerFlag )

FILE( GLOB filenames "cmake/Macros/*.cmake" )
FOREACH( filename ${filenames} )
  INCLUDE( ${filename} )
ENDFOREACH()

########################################################################
# Library setup
########################################################################

SET( PACKAGE_VERSION "0.1.1" )

FIND_PACKAGE( pybind11 REQUIRED )

# Searches for the Python libraries and the Python interpreter with the
# desired version.
#
# Note that the current development version has replace these functions
# by `FindPython3`. It is unclear when this feature will be rolled out,
# though. See
#
#   https://cmake.org/cmake/help/git-master/module/FindPython3.html
#
# for more information.



FIND_PACKAGE(PythonInterp 3 REQUIRED)
FIND_PACKAGE(PythonLibs   3 REQUIRED)


#IF( PYTHONINTERP_FOUND AND PYTHONLIBS_FOUND AND PYBIND11_FOUND )
  INCLUDE_DIRECTORIES(
    ${PYTHON_INCLUDE_DIRS}
    ${PYBIND11_INCLUDE_DIRS}
  )

message("Compilation starts")

ADD_LIBRARY(BAM_Feature_Store SHARED gids_nvme.cu )
 
  target_include_directories(BAM_Feature_Store PRIVATE ./include ../bam/include ../bam/include/freestanding/include/)
  target_link_libraries(BAM_Feature_Store PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../bam/build/lib/libnvm.so)
  # The library must not have any prefix and should be located in
  # a subfolder that includes the package name. The setup will be
  # more complicated otherwise.
  SET_TARGET_PROPERTIES( BAM_Feature_Store
    PROPERTIES
      PREFIX ""
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/BAM_Feature_Store"
  )

target_compile_options(BAM_Feature_Store PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
			-std=c++11;
					-O3;

					#				-G;
					#		-g;
			--default-stream per-thread;
			#	--ptxas-options=-v;
			#  	--compiler-options='-fPIC $(python3 -m pybind11 --includes)'
                      >)

target_compile_options(BAM_Feature_Store PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode arch=compute_70,code=sm_70>"
)
target_compile_options(BAM_Feature_Store PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode arch=compute_80,code=sm_80>"
)
  # Place the initialization file in the output directory for the Python
  # bindings. This will simplify the installation.
  CONFIGURE_FILE( BAM_Feature_Store/__init__.py
    ${CMAKE_CURRENT_BINARY_DIR}/BAM_Feature_Store/__init__.py
  )

  # Ditto for the setup file.
  CONFIGURE_FILE( BAM_Feature_Store/setup.py
    ${CMAKE_CURRENT_BINARY_DIR}/BAM_Feature_Store/setup.py
  )

#ENDIF()

