#ifndef __UTILS_H__
#define __UTILS_H__

#include <cmath>
#include <cstring>

/****************************************************/

bool is_fequal(float a, float b)
{
	return (fabs(a-b) < 0.00001);
}

bool is_equal(P_DATA_TYPE a, P_DATA_TYPE b)
{
	return ( is_fequal(a.x,b.x) && is_fequal(a.y,b.y) && is_fequal(a.z,b.z) && is_fequal(a.w,b.w) );
}

/*int get_grid_size(int n, int threadsPerBlock)
{
    int blocksPerGrid = (n+threadsPerBlock-1)/threadsPerBlock;
    return blocksPerGrid;
}

int get_block_size(int n)
{
    int threadsPerBlock;

    if(n <= WARP_SIZE) {
		threadsPerBlock = 2;
    }
    else {
	    if(n <= T_FACTOR*BLOCK_SIZE) {
		threadsPerBlock = WARP_SIZE;
	    }
	    else {
		threadsPerBlock = BLOCK_SIZE;
	    }
    }

    return threadsPerBlock;
}*/

int get_tpb(int n)
{
    int threadsPerBlock;

    if(n <= T_FACTOR*BLOCK_SIZE) {
	threadsPerBlock = WARP_SIZE;
    }
    else {
	threadsPerBlock = BLOCK_SIZE;
    }

    return threadsPerBlock;
}

/****************************************************/

int get_max_ChunkSize(INT4* hostChunktable) 
{
	int max_size = 0;
	for(int i=0; i<NUM_CHUNKS; i++) {
		int size = hostChunktable[i].c;
		if(size > max_size) max_size = size;
	}
	return max_size;
}

/****************************************************/

#endif

