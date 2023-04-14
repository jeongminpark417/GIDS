#include <pybind11/pybind11.h>
#include "cuda_lib.h"

namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}

void set_val(int64_t int_ptr, float a, int num){
	float* ptr = (float*) int_ptr;
	for(int i = 0; i < num; i++){
		ptr[i] = a;
	}
}

int64_t c_malloc(int bytes){
	int8_t* ptr = (int8_t*) malloc(bytes);
	return (int64_t)ptr;
}

float read_ptr_val(int64_t i_ptr){
	float* ptr = (float*) i_ptr;
	return(ptr[0]);
}

void set_ptr_val(int64_t i_ptr, float v){

	float* ptr = (float*) i_ptr;
	ptr[0] = v;

}

void set_cuda_val(int64_t i_ptr, int idx, float v){
	Direct_Loader DL;
	DL.h_cuda_write( i_ptr, idx,v);	
}


PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("add", &add, "A function that adds two numbers");
    m.def("set_val", &set_val, "set val");
    m.def("c_malloc", &c_malloc, "malloc all in c++");
    m.def("read_ptr_val", &read_ptr_val, "read_ptr_val");
    m.def("set_ptr_val", &set_ptr_val, "set_ptr_val");
    m.def("set_cuda_val", &set_cuda_val, "set cuda val");
}




