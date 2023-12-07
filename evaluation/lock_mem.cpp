/* This contains the mmap calls. */
#include <sys/mman.h>
/* These are for error printing. */
#include <errno.h>
#include <string.h>
#include <stdarg.h>
/* This is for open. */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <iostream>


int main() {
        int fd;
        struct stat s;
        size_t size;

        const char* mapped;
	char c;

	//Locking 512GB CPU memory
        size_t mem_lock = (512LL ) * 1024LL * 1024 * 1024;
        char* mem_p = (char*) malloc(mem_lock);

        mapped = (char*) mmap (0, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        //int ret = mlock(mem_p, mem_lock);
        int ret = mlockall(MCL_CURRENT);
        std::cout << "ret: " << ret << std::endl;
        std::cout  << "Done locking! ";
        std::cin >> c;

        munlockall();//(mapped, size);
}

