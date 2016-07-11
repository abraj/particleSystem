
#include "commonAPI.h"

#include "particleSystem.h"
#include "utils.h"

#include <curand_kernel.h>	// #include curand.h

#include <iostream>
#include <random>
#include <algorithm>

//#include <time.h>


/**
 * param1 set to default value MAX_PARTICLES
 * param1 changed to passed parameter value
 * param1 used by only lMaxParticlesxx, which remains UNUSED
 */
#define READ_NON_COMMON_ARGS \
	int param1 = MAX_PARTICLES_NUM; \
	FETCH_INT_ARG(param1, pCommonArgs, argc, argv); \
	size_t lMaxParticlesxx = param1; \
	lMaxParticlesxx = lMaxParticlesxx; 		// to suppress warning [unused variable]

/****************************************************/

float get_random_number_h(float min_val, float max_val)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dist(0, 1);

	return ( min_val + dist(gen)*(max_val-min_val) );
}

FLOAT3 get_random_uvector_h()
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dist(0, 1);

	int randInt1 = (int)(dist(gen) * 100) - (100/2);
	int randInt2 = (int)(dist(gen) * 100) - (100/2);
	int randInt3 = (int)(dist(gen) * 100) - (100/2);

	FLOAT3 vec = {randInt1*1.0, randInt2*1.0, randInt3*1.0};

	float mag = sqrtf(vec.x * vec.x * 1.0 + vec.y * vec.y * 1.0 + vec.z * vec.z * 1.0);
	vec.x /= mag;
	vec.y /= mag;
	vec.z /= mag;

	return vec;
}

/****************************************************/

namespace particleSystem
{

char* gSystemInput;
char* gParallelOutput;
char* gSerialOutput;

/* variables for serial execution */

/* variables for parallel execution */
P_DATA_TYPE* hostParticles;	// an array containing information about particles
T_DATA_TYPE* hostTdata;		// an array containing temporary information about particles (position)
int* hostQueue;			// queues containing set of praticle ids available for creation (segment-wise)
QUEUE_INFO* hostQueueInfo;	// a list containing info for each queue
int* hostChunkgrid;		// chunk-wise list of particles
int* hostCellgrid;		// cell-wise list of particles
int* hostGridMax;		// 0: max count particles in a chunk; 1: max count particles in a cell
PAIR* hostPkgDistrib;		// contains distributed segments (seg_type, seg_tid) for a package
curandState* hostRandstates;	// cuda random states

size_t  nParticles, 	sizeParticles;
size_t  nTdata, 	sizeTdata;
size_t  nQueue, 	sizeQueue;
size_t  nQueueInfo, 	sizeQueueInfo;
size_t  nChunkgrid, 	sizeChunkgrid;
size_t  nCellgrid, 	sizeCellgrid;
size_t  nGridMax, 	sizeGridMax;
size_t  nPkgDistrib, 	sizePkgDistrib;
size_t  nRandstates,	sizeRandstates;

size_t size_mem_blocks;		// total size of memory blocks

/****************************************************/

pmMemHandle pm_mem_alloc(size_t size)
{
	pmMemHandle lmem_handle;
	CREATE_MEM(size, lmem_handle)
	return lmem_handle;
}

void pm_mem_free(pmMemHandle lmem_handle)
{
    pmRawMemPtr lRawMemPtr;
    pmGetRawMemPtr(lmem_handle, &lRawMemPtr);

    // DELETE_MEM(lRawMemPtr);
}

void pm_mem_copyin(pmMemHandle lmem_handle, void *mem_ptr, size_t size)
{
    pmRawMemPtr lRawMemPtr;
    pmGetRawMemPtr(lmem_handle, &lRawMemPtr);
    memcpy(lRawMemPtr, mem_ptr, size);
}

void pm_mem_copyin(pmMemHandle lmem_handle, void *mem_ptr, size_t shift, size_t size)
{
    pmRawMemPtr lRawMemPtr;
    pmGetRawMemPtr(lmem_handle, &lRawMemPtr);
    memcpy(((char*)lRawMemPtr)+shift, mem_ptr, size);
}

void pm_mem_copyout(void *mem_ptr, pmMemHandle lmem_handle, size_t size)
{
    pmRawMemPtr lRawMemPtr;
    pmGetRawMemPtr(lmem_handle, &lRawMemPtr);
    memcpy(mem_ptr, lRawMemPtr, size);
}

void pm_mem_copyout(void *mem_ptr, pmMemHandle lmem_handle, size_t shift, size_t size)
{
    pmRawMemPtr lRawMemPtr;
    pmGetRawMemPtr(lmem_handle, &lRawMemPtr);
    memcpy(mem_ptr, ((char*)lRawMemPtr)+shift, size);
}

/****************************************************/

#ifdef BUILD_CUDA
pmCudaLaunchConf GetCudaLaunchConf(int numElems)
{
	pmCudaLaunchConf lCudaLaunchConf;

	int t = get_tpb(numElems);
	lCudaLaunchConf.threadsX = t;
	lCudaLaunchConf.blocksX = (numElems+t-1)/t;

	return lCudaLaunchConf;
}
#endif

pmStatus particleSystemDataDistribution_00(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	HOST_HEADER

	pmSubscriptionInfo lSubscriptionInfo;

	int index_mem_particles = 0;
	int index_mem_tdata 	= 1;

	int num_frags = NUM_FRAGS;
	int frag_len = (CONTAINER_SIZE+num_frags-1)/num_frags;
	int idx_start = subtask_id * frag_len;
	int idx_end   = subtask_id * frag_len + frag_len -1;
	int frag_lenx = (idx_end < CONTAINER_SIZE) ? frag_len : (CONTAINER_SIZE - idx_start);

	/*WRITE*/
	lSubscriptionInfo.offset = idx_start * sizeof(P_DATA_TYPE);
	lSubscriptionInfo.length = frag_lenx * sizeof(P_DATA_TYPE);
	pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, index_mem_particles, WRITE_SUBSCRIPTION, lSubscriptionInfo);

	/*WRITE*/
	lSubscriptionInfo.offset = idx_start * sizeof(T_DATA_TYPE);
	lSubscriptionInfo.length = frag_lenx * sizeof(T_DATA_TYPE);
	pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, index_mem_tdata, WRITE_SUBSCRIPTION, lSubscriptionInfo);

#ifdef BUILD_CUDA
	// Set CUDA Launch Configuration
	if(pDeviceInfo.deviceType == pm::GPU_CUDA)
		pmSetCudaLaunchConf(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, lTaskConf->cudaLaunchConf);
#endif
	return pmSuccess;
}

pmStatus particleSystemDataDistribution_00_s(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	HOST_HEADER

	pmSubscriptionInfo lSubscriptionInfo;

	int index_mem_particles = 0;

	/*WRITE*/
	lSubscriptionInfo.offset = 0;
	lSubscriptionInfo.length = sizeParticles;
	pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, index_mem_particles, WRITE_SUBSCRIPTION, lSubscriptionInfo);

#ifdef BUILD_CUDA
	// Set CUDA Launch Configuration
	if(pDeviceInfo.deviceType == pm::GPU_CUDA)
		pmSetCudaLaunchConf(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, lTaskConf->cudaLaunchConf);
#endif
	return pmSuccess;
}

pmStatus particleSystemDataDistribution_05_s(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	HOST_HEADER

	pmSubscriptionInfo lSubscriptionInfo;

	int index_mem_queueinfo = 0;
	int index_mem_queue 	= 1;
	int index_mem_particles = 2;

	/*WRITE*/
	lSubscriptionInfo.offset = 0;
	lSubscriptionInfo.length = sizeQueueInfo;
	pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, index_mem_queueinfo, READ_WRITE_SUBSCRIPTION, lSubscriptionInfo);

	/*WRITE*/
	lSubscriptionInfo.offset = 0;
	lSubscriptionInfo.length = sizeQueue;
	pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, index_mem_queue, READ_WRITE_SUBSCRIPTION, lSubscriptionInfo);

	/*WRITE*/
	lSubscriptionInfo.offset = 0;
	lSubscriptionInfo.length = sizeParticles;
	pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, index_mem_particles, WRITE_SUBSCRIPTION, lSubscriptionInfo);

#ifdef BUILD_CUDA
	// Set CUDA Launch Configuration
	if(pDeviceInfo.deviceType == pm::GPU_CUDA)
		pmSetCudaLaunchConf(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, lTaskConf->cudaLaunchConf);
#endif
	return pmSuccess;
}

pmStatus particleSystemDataDistribution_01(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	HOST_HEADER

	pmSubscriptionInfo lSubscriptionInfo;

	int index_mem_randstates = 0;

	/*WRITE*/
	lSubscriptionInfo.offset = 0;
	lSubscriptionInfo.length = sizeRandstates;
	pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, index_mem_randstates, WRITE_SUBSCRIPTION, lSubscriptionInfo);

#ifdef BUILD_CUDA
	// Set CUDA Launch Configuration
	if(pDeviceInfo.deviceType == pm::GPU_CUDA)
		pmSetCudaLaunchConf(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, lTaskConf->cudaLaunchConf);
#endif
	return pmSuccess;
}

pmStatus particleSystemDataDistribution_02(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	HOST_HEADER

	pmSubscriptionInfo lSubscriptionInfo;

	int index_mem_queueinfo = 0;
	int index_mem_queue = 1;

	/*WRITE*/
	lSubscriptionInfo.offset = 0;
	lSubscriptionInfo.length = sizeQueueInfo;
	pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, index_mem_queueinfo, WRITE_SUBSCRIPTION, lSubscriptionInfo);

	/*WRITE*/
	lSubscriptionInfo.offset = 0;
	lSubscriptionInfo.length = sizeQueue;
	pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, index_mem_queue, WRITE_SUBSCRIPTION, lSubscriptionInfo);

#ifdef BUILD_CUDA
	// Set CUDA Launch Configuration
	if(pDeviceInfo.deviceType == pm::GPU_CUDA)
		pmSetCudaLaunchConf(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, lTaskConf->cudaLaunchConf);
#endif
	return pmSuccess;
}

pmStatus particleSystemDataDistribution_02_s(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	HOST_HEADER

	pmSubscriptionInfo lSubscriptionInfo;

	int index_mem_queueinfo = 0;
	int index_mem_queue = 1;

	/*WRITE*/
	lSubscriptionInfo.offset = 0;
	lSubscriptionInfo.length = sizeQueueInfo;
	pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, index_mem_queueinfo, WRITE_SUBSCRIPTION, lSubscriptionInfo);

	/*WRITE*/
	lSubscriptionInfo.offset = 0;
	lSubscriptionInfo.length = sizeQueue;
	pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, index_mem_queue, WRITE_SUBSCRIPTION, lSubscriptionInfo);

#ifdef BUILD_CUDA
	// Set CUDA Launch Configuration
	if(pDeviceInfo.deviceType == pm::GPU_CUDA)
		pmSetCudaLaunchConf(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, lTaskConf->cudaLaunchConf);
#endif
	return pmSuccess;
}

pmStatus particleSystemDataDistribution_03(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	HOST_HEADER

	pmSubscriptionInfo lSubscriptionInfo;

	int index_mem_chunkgrid = 0;
	int index_mem_cellgrid 	= 1;
	int index_mem_gridmax 	= 2;

	/*WRITE*/
	lSubscriptionInfo.offset = 0;
	lSubscriptionInfo.length = sizeChunkgrid;
	pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, index_mem_chunkgrid, WRITE_SUBSCRIPTION, lSubscriptionInfo);

	/*WRITE*/
	lSubscriptionInfo.offset = 0;
	lSubscriptionInfo.length = sizeCellgrid;
	pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, index_mem_cellgrid, WRITE_SUBSCRIPTION, lSubscriptionInfo);

	/*WRITE*/
	lSubscriptionInfo.offset = 0;
	lSubscriptionInfo.length = sizeGridMax;
	pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, index_mem_gridmax, WRITE_SUBSCRIPTION, lSubscriptionInfo);

#ifdef BUILD_CUDA
	// Set CUDA Launch Configuration
	if(pDeviceInfo.deviceType == pm::GPU_CUDA)
		pmSetCudaLaunchConf(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, lTaskConf->cudaLaunchConf);
#endif
	return pmSuccess;
}

pmStatus particleSystemDataDistribution_03_s(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	HOST_HEADER

	pmSubscriptionInfo lSubscriptionInfo;

	int index_mem_chunkgrid = 3;
	int index_mem_cellgrid 	= 4;
	int index_mem_gridmax 	= 5;

	/*WRITE*/
	lSubscriptionInfo.offset = 0;
	lSubscriptionInfo.length = sizeChunkgrid;
	pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, index_mem_chunkgrid, WRITE_SUBSCRIPTION, lSubscriptionInfo);

	/*WRITE*/
	lSubscriptionInfo.offset = 0;
	lSubscriptionInfo.length = sizeCellgrid;
	pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, index_mem_cellgrid, WRITE_SUBSCRIPTION, lSubscriptionInfo);

	/*WRITE*/
	lSubscriptionInfo.offset = 0;
	lSubscriptionInfo.length = sizeGridMax;
	pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, index_mem_gridmax, WRITE_SUBSCRIPTION, lSubscriptionInfo);

#ifdef BUILD_CUDA
	// Set CUDA Launch Configuration
	if(pDeviceInfo.deviceType == pm::GPU_CUDA)
		pmSetCudaLaunchConf(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, lTaskConf->cudaLaunchConf);
#endif
	return pmSuccess;
}

pmStatus particleSystemDataDistribution_06(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	HOST_HEADER

	pmSubscriptionInfo lSubscriptionInfo;

	int index_mem_chunkgrid	 = 0;
	int index_mem_cellgrid 	 = 1;
	int index_mem_pkgdistrib = 2;
	int index_mem_queueinfo  = 3;

	int index_mem_queue 	 = 4;
	int index_mem_particles  = 5;
	int index_mem_tdata  	 = 6;

	/*READ*/
	lSubscriptionInfo.offset = subtask_id * (1+MAX_PARTICLES_PER_CHUNK) * sizeof(int);
	lSubscriptionInfo.length = (1+MAX_PARTICLES_PER_CHUNK) * sizeof(int);
	pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, index_mem_chunkgrid, READ_SUBSCRIPTION, lSubscriptionInfo);

	INT3 p = get_chunkIndex(subtask_id);
	int i1 = p.a;
	int i2 = p.b;
	int i3 = p.c;

	for(int i=0; i<CHUNK_DIM; i++) {
		for(int j=0; j<CHUNK_DIM; j++) {
			int start = GRID_DIM*GRID_DIM*(i3*CHUNK_DIM+i) + GRID_DIM*(i1*CHUNK_DIM+j) + (i2*CHUNK_DIM);
			int len = CHUNK_DIM; 

			//READ
			lSubscriptionInfo.offset = start * (1+MAX_PARTICLES_PER_CELL) * sizeof(int);
			lSubscriptionInfo.length = len * (1+MAX_PARTICLES_PER_CELL) * sizeof(int);
			pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, index_mem_cellgrid, READ_SUBSCRIPTION, lSubscriptionInfo);
		}
	}

	//READ
	lSubscriptionInfo.offset = subtask_id * 27 * sizeof(PAIR);
	lSubscriptionInfo.length = 27 * sizeof(PAIR);
	pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, index_mem_pkgdistrib, READ_SUBSCRIPTION, lSubscriptionInfo);

	//READ-WRITE
	lSubscriptionInfo.offset = 0;
	lSubscriptionInfo.length = sizeQueueInfo;
	pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, index_mem_queueinfo, READ_WRITE_SUBSCRIPTION, lSubscriptionInfo);

	//***************************************

	PAIR *seg_list = (hostPkgDistrib + 27*subtask_id);

	for(int i=0; i<27; i++) {
		int seg_type = seg_list[i].c;
		int seg_tid = seg_list[i].p;

	//-----------------------

		//int frag_rloc = get_cont_rloc(seg_type, seg_tid);

		int pos = 0;

		switch(seg_type) {
			case 8: pos += SEG4_SIZE;
			case 4: pos += SEG2_SIZE;
			case 2: pos += SEG1_SIZE;
			case 1: break;
			default: break;
		}
		switch(seg_type) {
			case 8: pos += seg_tid * SEG8_SIZE_T; break;
			case 4: pos += seg_tid * SEG4_SIZE_T; break;
			case 2: pos += seg_tid * SEG2_SIZE_T; break;
			case 1: pos += seg_tid * SEG1_SIZE_T; break;
			default: break;
		}

		int frag_rloc = pos;

	//-----------------------

		int frag_len;
		switch(seg_type) {
			case 1: frag_len = SEG1_SIZE_T; break;
			case 2: frag_len = SEG2_SIZE_T; break;
			case 4: frag_len = SEG4_SIZE_T; break;
			case 8: frag_len = SEG8_SIZE_T; break;
			default: frag_len = 64; break;  // unreachable!
		}

		//printf("----> [%02d] %d %02d : %04d, %04d\n", subtask_id, seg_type, seg_tid, frag_rloc, frag_len);

		//READ-WRITE
		lSubscriptionInfo.offset = frag_rloc * sizeof(int);
		lSubscriptionInfo.length = frag_len * sizeof(int);
		pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, index_mem_queue, READ_WRITE_SUBSCRIPTION, lSubscriptionInfo);

		//READ-WRITE
		lSubscriptionInfo.offset = frag_rloc * sizeof(P_DATA_TYPE);
		lSubscriptionInfo.length = frag_len * sizeof(P_DATA_TYPE);
		pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, index_mem_particles, READ_WRITE_SUBSCRIPTION, lSubscriptionInfo);

		//READ
		lSubscriptionInfo.offset = frag_rloc * sizeof(T_DATA_TYPE);
		lSubscriptionInfo.length = frag_len * sizeof(T_DATA_TYPE);
		pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, index_mem_tdata, READ_SUBSCRIPTION, lSubscriptionInfo);
	}

	//***************************************

//lSubscriptionInfo.offset = 0;
//lSubscriptionInfo.length = sizeParticles;
//pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, index_mem_particles, READ_SUBSCRIPTION, lSubscriptionInfo);

#ifdef BUILD_CUDA
	// Set CUDA Launch Configuration
	if(pDeviceInfo.deviceType == pm::GPU_CUDA)
		pmSetCudaLaunchConf(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, lTaskConf->cudaLaunchConf);
#endif
	return pmSuccess;
}

pmStatus particleSystemDataDistribution_06_s(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	HOST_HEADER

	pmSubscriptionInfo lSubscriptionInfo;

	int index_mem_chunkgrid	 = 0;
	int index_mem_cellgrid 	 = 1;
	int index_mem_pkgdistrib = 2;
	int index_mem_particles  = 3;

	/*READ*/
	lSubscriptionInfo.offset = 0;
	lSubscriptionInfo.length = sizeChunkgrid;
	pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, index_mem_chunkgrid, READ_SUBSCRIPTION, lSubscriptionInfo);

	/*READ*/
	lSubscriptionInfo.offset = 0;
	lSubscriptionInfo.length = sizeCellgrid;
	pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, index_mem_cellgrid, READ_SUBSCRIPTION, lSubscriptionInfo);

	/*READ*/
	lSubscriptionInfo.offset = 0;
	lSubscriptionInfo.length = sizePkgDistrib;
	pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, index_mem_pkgdistrib, READ_SUBSCRIPTION, lSubscriptionInfo);

	/*READ*/
	lSubscriptionInfo.offset = 0;
	lSubscriptionInfo.length = sizeParticles;
	pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, index_mem_particles, READ_SUBSCRIPTION, lSubscriptionInfo);

#ifdef BUILD_CUDA
	// Set CUDA Launch Configuration
	if(pDeviceInfo.deviceType == pm::GPU_CUDA)
		pmSetCudaLaunchConf(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, lTaskConf->cudaLaunchConf);
#endif
	return pmSuccess;
}

pmStatus particleSystemDataDistribution_04(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	HOST_HEADER

	pmSubscriptionInfo lSubscriptionInfo;

	int index_mem_pkgdistrib = 0;

	/*WRITE*/
	lSubscriptionInfo.offset = 0;
	lSubscriptionInfo.length = sizePkgDistrib;
	pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, index_mem_pkgdistrib, WRITE_SUBSCRIPTION, lSubscriptionInfo);

#ifdef BUILD_CUDA
	// Set CUDA Launch Configuration
	if(pDeviceInfo.deviceType == pm::GPU_CUDA)
		pmSetCudaLaunchConf(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, lTaskConf->cudaLaunchConf);
#endif
	return pmSuccess;
}

pmStatus particleSystemDataDistribution_04_s(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	HOST_HEADER

	pmSubscriptionInfo lSubscriptionInfo;

	int index_mem_pkgdistrib = 0;

	/*WRITE*/
	lSubscriptionInfo.offset = 0;
	lSubscriptionInfo.length = sizePkgDistrib;
	pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, index_mem_pkgdistrib, WRITE_SUBSCRIPTION, lSubscriptionInfo);

#ifdef BUILD_CUDA
	// Set CUDA Launch Configuration
	if(pDeviceInfo.deviceType == pm::GPU_CUDA)
		pmSetCudaLaunchConf(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, lTaskConf->cudaLaunchConf);
#endif
	return pmSuccess;
}

pmStatus particleSystemDataDistribution_08(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	HOST_HEADER

	pmSubscriptionInfo lSubscriptionInfo;

	int index_mem_chunkgrid = 0;
	int index_mem_cellgrid 	= 1;
	int index_mem_gridmax 	= 2;
	int index_mem_queueinfo = 3;
	int index_mem_queue 	= 4;
	int index_mem_particles = 5;
	int index_mem_tdata 	= 6;

	/*WRITE*/
	lSubscriptionInfo.offset = 0;
	lSubscriptionInfo.length = sizeChunkgrid;
	pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, index_mem_chunkgrid, READ_WRITE_SUBSCRIPTION, lSubscriptionInfo);

	/*WRITE*/
	lSubscriptionInfo.offset = 0;
	lSubscriptionInfo.length = sizeCellgrid;
	pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, index_mem_cellgrid, READ_WRITE_SUBSCRIPTION, lSubscriptionInfo);

	/*WRITE*/
	lSubscriptionInfo.offset = 0;
	lSubscriptionInfo.length = sizeGridMax;
	pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, index_mem_gridmax, READ_WRITE_SUBSCRIPTION, lSubscriptionInfo);

	/*READ-WRITE*/
	lSubscriptionInfo.offset = 0;
	lSubscriptionInfo.length = sizeQueueInfo;
	pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, index_mem_queueinfo, READ_WRITE_SUBSCRIPTION, lSubscriptionInfo);

	/*READ-WRITE*/
	lSubscriptionInfo.offset = 0;
	lSubscriptionInfo.length = sizeQueue;
	pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, index_mem_queue, READ_WRITE_SUBSCRIPTION, lSubscriptionInfo);

	/*READ-WRITE*/
	lSubscriptionInfo.offset = 0;
	lSubscriptionInfo.length = sizeParticles;
	pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, index_mem_particles, READ_WRITE_SUBSCRIPTION, lSubscriptionInfo);

	/*WRITE*/
	lSubscriptionInfo.offset = 0;
	lSubscriptionInfo.length = sizeTdata;
	pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, index_mem_tdata, WRITE_SUBSCRIPTION, lSubscriptionInfo);

#ifdef BUILD_CUDA
	// Set CUDA Launch Configuration
	if(pDeviceInfo.deviceType == pm::GPU_CUDA)
		pmSetCudaLaunchConf(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, lTaskConf->cudaLaunchConf);
#endif
	return pmSuccess;
}

pmStatus particleSystemDataDistribution_08_s(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	HOST_HEADER

	pmSubscriptionInfo lSubscriptionInfo;

	int index_mem_chunkgrid = 0;
	int index_mem_cellgrid 	= 1;
	int index_mem_gridmax 	= 2;
	int index_mem_queueinfo = 3;
	int index_mem_queue 	= 4;
	int index_mem_particles = 5;

	/*WRITE*/
	lSubscriptionInfo.offset = 0;
	lSubscriptionInfo.length = sizeChunkgrid;
	pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, index_mem_chunkgrid, READ_WRITE_SUBSCRIPTION, lSubscriptionInfo);

	/*WRITE*/
	lSubscriptionInfo.offset = 0;
	lSubscriptionInfo.length = sizeCellgrid;
	pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, index_mem_cellgrid, READ_WRITE_SUBSCRIPTION, lSubscriptionInfo);

	/*WRITE*/
	lSubscriptionInfo.offset = 0;
	lSubscriptionInfo.length = sizeGridMax;
	pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, index_mem_gridmax, READ_WRITE_SUBSCRIPTION, lSubscriptionInfo);

	/*READ-WRITE*/
	lSubscriptionInfo.offset = 0;
	lSubscriptionInfo.length = sizeQueueInfo;
	pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, index_mem_queueinfo, READ_WRITE_SUBSCRIPTION, lSubscriptionInfo);

	/*READ-WRITE*/
	lSubscriptionInfo.offset = 0;
	lSubscriptionInfo.length = sizeQueue;
	pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, index_mem_queue, READ_WRITE_SUBSCRIPTION, lSubscriptionInfo);

	/*READ-WRITE*/
	lSubscriptionInfo.offset = 0;
	lSubscriptionInfo.length = sizeParticles;
	pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, index_mem_particles, READ_WRITE_SUBSCRIPTION, lSubscriptionInfo);

#ifdef BUILD_CUDA
	// Set CUDA Launch Configuration
	if(pDeviceInfo.deviceType == pm::GPU_CUDA)
		pmSetCudaLaunchConf(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, lTaskConf->cudaLaunchConf);
#endif
	return pmSuccess;
}

// NOT used
pmStatus particleSystem_cpu(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	return pmSuccess;
}

/******************************************************/

pmStatus particleSystem_init_particles_cpu(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	HOST_HEADER

	int index_mem_particles = 0;

	P_DATA_TYPE* devParticles = (P_DATA_TYPE*)(pSubtaskInfo.memInfo[index_mem_particles].ptr);

	for(int tid=0; tid<subtask_elems; tid++)
	{

	devParticles[tid].id = tid;
	init_particle(devParticles[tid]);

	}

	return pmSuccess;
}

pmStatus particleSystem_init_particles_host(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	HOST_HEADER

	int index_mem_particles = 0;
	int index_mem_tdata 	= 1;

	P_DATA_TYPE* devParticles = (P_DATA_TYPE*)(pSubtaskInfo.memInfo[index_mem_particles].ptr);
	T_DATA_TYPE* devTdata = (T_DATA_TYPE*)(pSubtaskInfo.memInfo[index_mem_tdata].ptr);

	for(int tid=0; tid<subtask_elems; tid++)
	{

	int num_frags = NUM_FRAGS;
	int frag_len = (CONTAINER_SIZE+num_frags-1)/num_frags;
	int gid = subtask_id * frag_len + tid;
	if(gid > CONTAINER_SIZE-1) continue;

	devParticles[tid].id = gid;
	init_particle(devParticles[tid]);

	devTdata[tid].id = gid;
	devTdata[tid].x = 0.0f;
	devTdata[tid].y = 0.0f;
	devTdata[tid].z = 0.0f;
	devTdata[tid].w = 0.0f;
	devTdata[tid].age = 0.0f;

	}

	return pmSuccess;
}

pmStatus particleSystem_q_start_fast_cpu(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	HOST_HEADER

	int seg_type, seg_tid;
	int rloc, seg_size;

	int index_mem_queueinfo = 0;
	int index_mem_queue = 1;

	QUEUE_INFO* devQueueInfo = (QUEUE_INFO*)(pSubtaskInfo.memInfo[index_mem_queueinfo].ptr);
	int* devQueue = (int*)(pSubtaskInfo.memInfo[index_mem_queue].ptr);

	for(int tid=0; tid<subtask_elems; tid++)
	{

	devQueue[tid] = tid;

	if(tid < QUEUE_INFO_SIZE) {
		if(tid >= SEG1_COUNT + SEG2_COUNT + SEG4_COUNT) {
			seg_type = 8;
			seg_tid = tid - (SEG1_COUNT + SEG2_COUNT + SEG4_COUNT);
		}
		else if(tid >= SEG1_COUNT + SEG2_COUNT) {
			seg_type = 4;
			seg_tid = tid - (SEG1_COUNT + SEG2_COUNT);
		}
		else if(tid >= SEG1_COUNT) {
			seg_type = 2;
			seg_tid = tid - SEG1_COUNT;
		}
		else {
			seg_type = 1;
			seg_tid = tid;
		}

		switch(seg_type) {
			case 8: seg_size = SEG8_SIZE_T; break;
			case 4: seg_size = SEG4_SIZE_T; break;
			case 2: seg_size = SEG2_SIZE_T; break;
			case 1: seg_size = SEG1_SIZE_T; break;
			default: seg_size = 64; break; // unreachable!
		}

		rloc = get_cont_rloc(seg_type, seg_tid);

		devQueueInfo[tid].front = rloc;
		devQueueInfo[tid].rear = rloc+seg_size-1;
		devQueueInfo[tid].count = seg_size;
		devQueueInfo[tid].lock = 0;
		devQueueInfo[tid].rloc = rloc;
		devQueueInfo[tid].seg_size = seg_size;
	}

	}

	return pmSuccess;
}

pmStatus particleSystem_q_start_fast_host(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	HOST_HEADER

	int seg_type, seg_tid;
	int rloc, seg_size;

	int index_mem_queueinfo = 0;
	int index_mem_queue = 1;

	QUEUE_INFO* devQueueInfo = (QUEUE_INFO*)(pSubtaskInfo.memInfo[index_mem_queueinfo].ptr);
	int* devQueue = (int*)(pSubtaskInfo.memInfo[index_mem_queue].ptr);

	for(int tid=0; tid<subtask_elems; tid++)
	{

	devQueue[tid] = tid;

	if(tid < QUEUE_INFO_SIZE) {
		if(tid >= SEG1_COUNT + SEG2_COUNT + SEG4_COUNT) {
			seg_type = 8;
			seg_tid = tid - (SEG1_COUNT + SEG2_COUNT + SEG4_COUNT);
		}
		else if(tid >= SEG1_COUNT + SEG2_COUNT) {
			seg_type = 4;
			seg_tid = tid - (SEG1_COUNT + SEG2_COUNT);
		}
		else if(tid >= SEG1_COUNT) {
			seg_type = 2;
			seg_tid = tid - SEG1_COUNT;
		}
		else {
			seg_type = 1;
			seg_tid = tid;
		}

		switch(seg_type) {
			case 8: seg_size = SEG8_SIZE_T; break;
			case 4: seg_size = SEG4_SIZE_T; break;
			case 2: seg_size = SEG2_SIZE_T; break;
			case 1: seg_size = SEG1_SIZE_T; break;
			default: seg_size = 64; break; // unreachable!
		}

		rloc = get_cont_rloc(seg_type, seg_tid);

		devQueueInfo[tid].front = rloc;
		devQueueInfo[tid].rear = rloc+seg_size-1;
		devQueueInfo[tid].count = seg_size;
		devQueueInfo[tid].lock = 0;
		devQueueInfo[tid].rloc = rloc;
		devQueueInfo[tid].seg_size = seg_size;
	}

	}

	return pmSuccess;
}

pmStatus particleSystem_pkg_distrib_cpu(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	HOST_HEADER

	int index_mem_pkgdistrib = 0;

	PAIR* devPkgDistrib = (PAIR*)(pSubtaskInfo.memInfo[index_mem_pkgdistrib].ptr);

	for(int tid=0; tid<subtask_elems; tid++)
	{
		if(tid >= NUM_CHUNKS) continue;

		int chunk = tid;
		PAIR *seg_list = devPkgDistrib + (tid*27);
		set_pkg_segments(chunk, seg_list);
	}

	return pmSuccess;
}

pmStatus particleSystem_pkg_distrib_host(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	HOST_HEADER

	int index_mem_pkgdistrib = 0;

	PAIR* devPkgDistrib = (PAIR*)(pSubtaskInfo.memInfo[index_mem_pkgdistrib].ptr);

	for(int tid=0; tid<subtask_elems; tid++)
	{
		if(tid >= NUM_CHUNKS) continue;

		int chunk = tid;
		PAIR *seg_list = devPkgDistrib + (tid*27);
		set_pkg_segments(chunk, seg_list);
	}

	return pmSuccess;
}

/******************************************************/

int fill_particle(P_DATA_TYPE* devParticles, QUEUE_INFO* devQueueInfo, int* devQueue, float x, float y, float z)
{
	int i1, i2, i3;
	int cell, seg_type, seg_tid;
	int nid = -1;

	i1 = floor((-1.0 * y) / CELL_SIZE) + (GRID_DIM/2);
	i2 = floor(( 1.0 * x) / CELL_SIZE) + (GRID_DIM/2);
	i3 = floor((-1.0 * z) / CELL_SIZE) + (GRID_DIM/2);

	if( (i1 >= 0 && i1 < GRID_DIM) && (i2 >= 0 && i2 < GRID_DIM) && (i3 >= 0 && i3 < GRID_DIM) ) {

		cell = i3*GRID_DIM*GRID_DIM + i1*GRID_DIM + i2;

		INT3 cell_info = get_cell_info(cell);
		//chunk    = cell_info.a;
		seg_type = cell_info.b;
		seg_tid  = cell_info.c;

		nid = q_remove(devQueueInfo, devQueue, seg_type, seg_tid);

		if(nid < 0) {
			printf("ERROR! Overflow (Reserved space full).\n");
			exit(1);
		}
		else {
			//std::mt19937::result_type seed = time(0);
			//std::mt19937 gen(seed);

			std::random_device rd;
			std::mt19937 gen(rd());
			std::uniform_real_distribution<> dist(0, 1);

			float w = PARTICLE_WEIGHT_DEFAULT;
			float age = MIN_ADULT_AGE + dist(gen)*(MAX_ADULT_AGE-MIN_ADULT_AGE);
			float fert_age = MIN_FERTILITY_AGE + dist(gen)*(MAX_FERTILITY_AGE-MIN_FERTILITY_AGE);
			create_particle_s(devParticles, nid, w, age, fert_age, x, y, z, 0.0, 0.0, 0.0);
		}
	}
	else {
		printf("ERROR! Particle location OUTSIDE box.\n");
		exit(1);
	}

	return nid;
}

pmStatus particleSystem_fill_particles_cpu(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	HOST_HEADER

	int index_mem_queueinfo = 0;
	int index_mem_queue 	= 1;
	int index_mem_particles = 2;

	QUEUE_INFO* devQueueInfo = (QUEUE_INFO*)(pSubtaskInfo.memInfo[index_mem_queueinfo].ptr);
	int* devQueue = (int*)(pSubtaskInfo.memInfo[index_mem_queue].ptr);
	P_DATA_TYPE* devParticles = (P_DATA_TYPE*)(pSubtaskInfo.memInfo[index_mem_particles].ptr);

	double range = ((GRID_DIM)/2)*CELL_SIZE;

//	srand (time(NULL));  // initialize random seed
//	int sign = (rand() % 2 == 0) ? 1 : -1;
//	double r = ((double) rand() / (RAND_MAX));

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dist(0, 1);

	double r, sign;
	float x, y, z;

	for (int i = 0; i < MAX_PARTICLES_NUM; i++) {
		r = dist(gen);
		sign = (dist(gen) >= 0.5) ? 1 : -1;
		x = (float)sign*r*range;

		r = dist(gen);
		sign = (dist(gen) >= 0.5) ? 1 : -1;
		y = (float)sign*r*range;

		r = dist(gen);
		sign = (dist(gen) >= 0.5) ? 1 : -1;
		z = (float)sign*r*range;

		//printf("=======> %5.2f %5.2f %5.2f\n", x, y, z);

		/************************

		int i1, i2, i3;
		int chunk = -1;

		i1 = floor((-1.0 * y) / CELL_SIZE) + (GRID_DIM/2);
		i2 = floor(( 1.0 * x) / CELL_SIZE) + (GRID_DIM/2);
		i3 = floor((-1.0 * z) / CELL_SIZE) + (GRID_DIM/2);

		if( (i1 >= 0 && i1 < GRID_DIM) && (i2 >= 0 && i2 < GRID_DIM) && (i3 >= 0 && i3 < GRID_DIM) ) {

			int cell = i3*GRID_DIM*GRID_DIM + i1*GRID_DIM + i2;

			INT3 cell_info = get_cell_info(cell);
			chunk    = cell_info.a;
		}

		if(chunk == 0) {
			fill_particle(devParticles, devQueueInfo, devQueue, x, y, z);
		}
		else {
			i--;
		}

		/************************/

		fill_particle(devParticles, devQueueInfo, devQueue, x, y, z);
	}

//	int nid = fill_particle(devParticles, devQueueInfo, devQueue,  0.0f, 0.0f, 4.0f);  // particle 0

//	int id1 = fill_particle(devParticles, devQueueInfo, devQueue, -4.0f, 0.0f, 0.0f);  // particle 0
//	int id2 = fill_particle(devParticles, devQueueInfo, devQueue,  4.0f, 0.0f, 0.0f);  // particle 1

//	fill_particle(devParticles, devQueueInfo, devQueue,  4.5f, 1.0f, 0.0f);  // particle 0
//	fill_particle(devParticles, devQueueInfo, devQueue,  9.5f, 1.0f, 0.0f);  // particle 1
//	fill_particle(devParticles, devQueueInfo, devQueue,  7.0f, 1.0f, 0.0f);  // particle 2

//	fill_particle(devParticles, devQueueInfo, devQueue, -1.0f, -4.0f, 0.0f);  // particle 0
//	fill_particle(devParticles, devQueueInfo, devQueue,  4.0f,  1.0f, 0.0f);  // particle 1
//	fill_particle(devParticles, devQueueInfo, devQueue,  1.0f,  4.0f, 0.0f);  // particle 2
//	fill_particle(devParticles, devQueueInfo, devQueue, -6.0f,  1.0f, 0.0f);  // particle 3
//	fill_particle(devParticles, devQueueInfo, devQueue,  4.0f,  4.0f, 0.0f);  // particle 4
//	fill_particle(devParticles, devQueueInfo, devQueue, -7.0f,  4.0f, 0.0f);  // particle 5

	return pmSuccess;
}

pmStatus particleSystem_calc_forces_cpu(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	HOST_HEADER

	int index_mem_chunkgrid	 = 1;
	int index_mem_cellgrid 	 = 2;
	int index_mem_pkgdistrib = 3;
	int index_mem_particles  = 5;

	int* devChunkgrid 	  = (int*)(pSubtaskInfo.memInfo[index_mem_chunkgrid].ptr);
	int* devCellgrid 	  = (int*)(pSubtaskInfo.memInfo[index_mem_cellgrid].ptr);
	PAIR* devPkgDistrib 	  = (PAIR*)(pSubtaskInfo.memInfo[index_mem_pkgdistrib].ptr);
	P_DATA_TYPE* devParticles = (P_DATA_TYPE*)(pSubtaskInfo.memInfo[index_mem_particles].ptr);

	// subtask_id exists even for SERIAL_MODE (control by setting ST_MAX to 1)

	for(int tid=0; tid<subtask_elems; tid++)
	{

	int c = (1+MAX_PARTICLES_PER_CHUNK);
	int i = subtask_id;
	int chunk_size = devChunkgrid[i*c+0];
	if(tid > chunk_size-1) continue;

	int pos;
	int nid, n_pos;

	FLOAT3 acc = {0.0f, 0.0f, 0.0f};

	PAIR *seg_list = devPkgDistrib;

	int pid = devChunkgrid[i*c+(tid+1)];

	if(SERIAL_MODE == 1) pos = pid;
	else pos = get_local_pos(pid, seg_list);

	if(pos == -1) continue;  // Should NEVER happen!

	P_DATA_TYPE& myParticle = devParticles[pos];

	if(myParticle.cell >= 0 && myParticle.cell < NUM_CELLS) {

		NEIB_CELLS neibCells;
		neibCells.size = 0;
		neibCells.data[neibCells.size++] = myParticle.cell;
		fill_cells(neibCells);

		NEIB_PARTICLES neibParticles;
		neibParticles.size = 0;
		fill_particles(neibParticles, neibCells, devCellgrid);

		for (int i = 0; i < neibParticles.size; i++) {

			nid = neibParticles.data[i];

			if(SERIAL_MODE == 1) n_pos = nid;
			else n_pos = get_local_pos(nid, seg_list);

			if(n_pos == -1) continue;  // Should NEVER happen!

			P_DATA_TYPE neibParticle = devParticles[n_pos];
			//if(myParticle.id != neibParticle.id) acc = bodyBodyInteraction(myParticle, neibParticle, acc);
		}
	}

	}

	return pmSuccess;
}

pmStatus particleSystem_calc_forces_host(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	HOST_HEADER

	int index_mem_chunkgrid	 = 0;
	int index_mem_cellgrid 	 = 1;
	int index_mem_pkgdistrib = 2;
	int index_mem_queueinfo  = 3;
	int index_mem_queue 	 = 4;
	int index_mem_particles  = 5;
	int index_mem_tdata  	 = 6;

	int* devChunkgrid 	  = (int*)(pSubtaskInfo.memInfo[index_mem_chunkgrid].ptr);
	int* devCellgrid 	  = (int*)(pSubtaskInfo.memInfo[index_mem_cellgrid].ptr);
	PAIR* devPkgDistrib 	  = (PAIR*)(pSubtaskInfo.memInfo[index_mem_pkgdistrib].ptr);
	QUEUE_INFO* devQueueInfo   = (QUEUE_INFO*)(pSubtaskInfo.memInfo[index_mem_queueinfo].ptr);
	int* devQueue 		   = (int*)(pSubtaskInfo.memInfo[index_mem_queue].ptr);
	P_DATA_TYPE* devParticles = (P_DATA_TYPE*)(pSubtaskInfo.memInfo[index_mem_particles].ptr);
	T_DATA_TYPE* devTdata = (T_DATA_TYPE*)(pSubtaskInfo.memInfo[index_mem_tdata].ptr);

	for(int tid=0; tid<subtask_elems; tid++)
	{

	//int c = (1+MAX_PARTICLES_PER_CHUNK);
	//int i = subtask_id;
	int chunk_size = devChunkgrid[0];
	if(tid > chunk_size-1) continue;

	int pos;

	FLOAT3 acc = {0.0f, 0.0f, 0.0f};
	int collision_flag = 0;

	PAIR *seg_list = devPkgDistrib;

	int pid = devChunkgrid[tid+1];

	if(SERIAL_MODE == 1) pos = pid;
	else if(HOST_MODE == 1) pos = get_natural_pos(pid, subtask_id);
	else pos = get_local_pos(pid, seg_list);

	if(pos == -1) continue;  // Should NEVER happen!

	P_DATA_TYPE& myParticle = devParticles[pos];

	int id = myParticle.id;
	int seg_type, seg_tid;
	int nid, n_pos;

	if(myParticle.cell >= 0 && myParticle.cell < NUM_CELLS) {

		NEIB_CELLS neibCells;
		neibCells.size = 0;
		neibCells.data[neibCells.size++] = myParticle.cell;
		fill_cells(neibCells);

		NEIB_PARTICLES neibParticles;
		neibParticles.size = 0;
		fill_particles(neibParticles, neibCells, devCellgrid, subtask_id);

		//*---------------------------------------

		// death
		if(myParticle.age > PARTICLE_LIFE) {
			collision_flag = 2;
		}

		// collision
		if(collision_flag == 0) {
		  for (int i = 0; i < neibParticles.size; i++) {

			int flag = 0;

			nid = neibParticles.data[i];

			if(SERIAL_MODE == 1) n_pos = nid;
			else if(HOST_MODE == 1) n_pos = get_natural_pos(nid, subtask_id);
			else n_pos = get_local_pos(nid, seg_list);

			if(n_pos == -1) continue;  // Should NEVER happen!

			T_DATA_TYPE neibTdata = devTdata[n_pos];
			if(myParticle.id != neibTdata.id) flag = bodyBodyCollision(myParticle, neibTdata);

			if(flag > collision_flag) collision_flag = flag;

			if(collision_flag == 2) break;
		  }
		}

		// kill
		if(collision_flag == 2) {

			if(id >= SEG1_SIZE + SEG2_SIZE + SEG4_SIZE) {
				seg_type = 8;
				seg_tid = (id - (SEG1_SIZE + SEG2_SIZE + SEG4_SIZE)) / SEG8_SIZE_T;
			}
			else if(id >= SEG1_SIZE + SEG2_SIZE) {
				seg_type = 4;
				seg_tid = (id - (SEG1_SIZE + SEG2_SIZE)) / SEG4_SIZE_T;
			}
			else if(id >= SEG1_SIZE) {
				seg_type = 2;
				seg_tid = (id - SEG1_SIZE) / SEG2_SIZE_T;
			}
			else {
				seg_type = 1;
				seg_tid = id / SEG1_SIZE_T;
			}

			reset_particle(myParticle);

			//lock-free (serial) !!!!!!!!
			// std::atomic_exchange, atomic_compare_exchange_..
			q_insert(devQueueInfo, devQueue, seg_type, seg_tid, id, subtask_id);
		}
		// survive
		else if(collision_flag == 1) {

			survive_particle(myParticle);
		}

		if(collision_flag > 0) continue;

		//*---------------------------------------

		// calculate acceleration
		for (int i = 0; i < neibParticles.size; i++) {

			nid = neibParticles.data[i];

			if(SERIAL_MODE == 1) n_pos = nid;
			else if(HOST_MODE == 1) n_pos = get_natural_pos(nid, subtask_id);
			else n_pos = get_local_pos(nid, seg_list);

			if(n_pos == -1) continue;  // Should NEVER happen!

			T_DATA_TYPE neibTdata = devTdata[n_pos];
			if(myParticle.id != neibTdata.id) acc = bodyBodyInteraction(myParticle, neibTdata, acc);
		}

		myParticle.ax = acc.x;
		myParticle.ay = acc.y;
		myParticle.az = acc.z;

		//*---------------------------------------

		// update position and velocity
		FLOAT3 r;
		float dx, dy, dz;
		float vx, vy, vz;

		float t = DT;

		dx = myParticle.vx * t + 0.5 * myParticle.ax * t * t;
		dy = myParticle.vy * t + 0.5 * myParticle.ay * t * t;
		dz = myParticle.vz * t + 0.5 * myParticle.az * t * t;

		float dmaxr = MAX_DX;
		if(std::abs(dx) > dmaxr) dx = dmaxr*(dx/std::abs(dx));
		if(std::abs(dy) > dmaxr) dy = dmaxr*(dy/std::abs(dy));
		if(std::abs(dz) > dmaxr) dz = dmaxr*(dz/std::abs(dz));

		r.x = myParticle.x + dx;
		r.y = myParticle.y + dy;
		r.z = myParticle.z + dz;

		set_pos_x(myParticle, r);

		vx = myParticle.vx + myParticle.ax * t;
		vy = myParticle.vy + myParticle.ay * t;
		vz = myParticle.vz + myParticle.az * t;

		float maxv = MAX_V;
		if(std::abs(vx) > maxv) vx = maxv*(vx/std::abs(vx));
		if(std::abs(vy) > maxv) vy = maxv*(vy/std::abs(vy));
		if(std::abs(vz) > maxv) vz = maxv*(vz/std::abs(vz));

		myParticle.vx = vx;
		myParticle.vy = vy;
		myParticle.vz = vz;

		myParticle.age += t;

		//*---------------------------------------

		// particle explosion
		if( (myParticle.age >= myParticle.fertility_age) && !(myParticle.is_parent) ) {

			FLOAT3 uvec = get_random_uvector_h();
			float vx = uvec.x * EXPLOSION_SPEED;
			float vy = uvec.y * EXPLOSION_SPEED;
			float vz = uvec.z * EXPLOSION_SPEED;

			myParticle.is_parent = true;
			myParticle.vx = vx;
			myParticle.vy = vy;
			myParticle.vz = vz;

			//lock-free (serial) !!!!!!!!
			// std::atomic_exchange, atomic_compare_exchange_..
			nid = q_remove(devQueueInfo, devQueue, myParticle.seg_type, myParticle.seg_tid, subtask_id);
			if(nid >= 0) {
				if(SERIAL_MODE == 1) n_pos = nid;
				else if(HOST_MODE == 1) n_pos = get_natural_pos(nid, subtask_id);
				else n_pos = get_local_pos(nid, seg_list);

				float w = PARTICLE_WEIGHT_DEFAULT;
				float age = 0.0f;
				float fert_age = get_random_number_h(MIN_FERTILITY_AGE, MAX_FERTILITY_AGE);

				create_particle_s(devParticles, n_pos, w, age, fert_age, myParticle.x, myParticle.y, myParticle.z, -1.0*vx, -1.0*vy, -1.0*vz);
			}
		}

		// change particle id on segment change
		if(myParticle.seg_fault) {

			if(id >= SEG1_SIZE + SEG2_SIZE + SEG4_SIZE) {
				seg_type = 8;
				seg_tid = (id - (SEG1_SIZE + SEG2_SIZE + SEG4_SIZE)) / SEG8_SIZE_T;
			}
			else if(id >= SEG1_SIZE + SEG2_SIZE) {
				seg_type = 4;
				seg_tid = (id - (SEG1_SIZE + SEG2_SIZE)) / SEG4_SIZE_T;
			}
			else if(id >= SEG1_SIZE) {
				seg_type = 2;
				seg_tid = (id - SEG1_SIZE) / SEG2_SIZE_T;
			}
			else {
				seg_type = 1;
				seg_tid = id / SEG1_SIZE_T;
			}

			//lock-free (serial) !!!!!!!!
			// std::atomic_exchange, atomic_compare_exchange_..
			nid = q_remove(devQueueInfo, devQueue, myParticle.seg_type, myParticle.seg_tid, subtask_id);
			if(nid >= 0) {
				if(SERIAL_MODE == 1) n_pos = nid;
				else if(HOST_MODE == 1) n_pos = get_natural_pos(nid, subtask_id);
				else n_pos = get_local_pos(nid, seg_list);

				copy_particle(devParticles[n_pos], myParticle);
				devParticles[n_pos].seg_fault = false;
			}

			reset_particle(myParticle);

			//lock-free (serial) !!!!!!!!
			// std::atomic_exchange, atomic_compare_exchange_..
			q_insert(devQueueInfo, devQueue, seg_type, seg_tid, id, subtask_id);

			continue;
		}

		//*---------------------------------------

	}

	}

	return pmSuccess;
}

/***********************************/

int serialAdd(int* address, int val)
{
	int old = *address;

	*address = old + val;

	return old;
}

int serialMax(int* address, int val)
{
	int old = *address;
	
	int max = std::max(old, val);
	*address = max;

	return old;
}

pmStatus particleSystem_build_grid_cpu(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	HOST_HEADER

	int index_mem_chunkgrid = 0;
	int index_mem_cellgrid 	= 1;
	int index_mem_gridmax 	= 2;
	int index_mem_queueinfo = 3;
	int index_mem_queue 	= 4;
	int index_mem_particles = 5;

	int* devChunkgrid = (int*)(pSubtaskInfo.memInfo[index_mem_chunkgrid].ptr);
	int* devCellgrid = (int*)(pSubtaskInfo.memInfo[index_mem_cellgrid].ptr);
	int* devGridMax = (int*)(pSubtaskInfo.memInfo[index_mem_gridmax].ptr);
	QUEUE_INFO* devQueueInfo = (QUEUE_INFO*)(pSubtaskInfo.memInfo[index_mem_queueinfo].ptr);
	int* devQueue = (int*)(pSubtaskInfo.memInfo[index_mem_queue].ptr);
	P_DATA_TYPE* devParticles = (P_DATA_TYPE*)(pSubtaskInfo.memInfo[index_mem_particles].ptr);

	for(int tid=0; tid<subtask_elems; tid++)
	{

	if(devParticles[tid].cell >= 0 && devParticles[tid].cell < NUM_CELLS) {

		int c, i, old;

		c = (1+MAX_PARTICLES_PER_CHUNK);
		i = devParticles[tid].chunk;
		old = serialAdd(&devChunkgrid[i*c+0], 1);
		if(old < MAX_PARTICLES_PER_CHUNK) {
			devChunkgrid[i*c+(old+1)] = devParticles[tid].id;
			serialMax(&devGridMax[0], old+1);
		}

		c = (1+MAX_PARTICLES_PER_CELL);
		i = devParticles[tid].cell;
		old = serialAdd(&devCellgrid[i*c+0], 1);
		if(old < MAX_PARTICLES_PER_CELL) {
			devCellgrid[i*c+(old+1)] = devParticles[tid].id;
			serialMax(&devGridMax[1], old+1);
		}
		else {
			serialAdd(&devCellgrid[i*c+0], -1);
			//printf("ERROR! MAX_PARTICLES_PER_CELL(%d) Overflow.\n", MAX_PARTICLES_PER_CELL);
			//printf(".");

			//kill
			reset_particle(devParticles[tid]);

			//lock-free (serial)
			q_insert(devQueueInfo, devQueue, devParticles[tid].seg_type, devParticles[tid].seg_tid, devParticles[tid].id);

			//exit(1);
			//asm("trap;"); // kill all threads
			//asm("exit;"); // kill this thread only
		}
	}

	}

	return pmSuccess;
}

pmStatus particleSystem_build_grid_host(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	HOST_HEADER

	int index_mem_chunkgrid = 0;
	int index_mem_cellgrid 	= 1;
	int index_mem_gridmax 	= 2;
	int index_mem_queueinfo = 3;
	int index_mem_queue 	= 4;
	int index_mem_particles = 5;
	int index_mem_tdata 	= 6;

	int* devChunkgrid = (int*)(pSubtaskInfo.memInfo[index_mem_chunkgrid].ptr);
	int* devCellgrid = (int*)(pSubtaskInfo.memInfo[index_mem_cellgrid].ptr);
	int* devGridMax = (int*)(pSubtaskInfo.memInfo[index_mem_gridmax].ptr);
	QUEUE_INFO* devQueueInfo = (QUEUE_INFO*)(pSubtaskInfo.memInfo[index_mem_queueinfo].ptr);
	int* devQueue = (int*)(pSubtaskInfo.memInfo[index_mem_queue].ptr);
	P_DATA_TYPE* devParticles = (P_DATA_TYPE*)(pSubtaskInfo.memInfo[index_mem_particles].ptr);
	T_DATA_TYPE* devTdata = (T_DATA_TYPE*)(pSubtaskInfo.memInfo[index_mem_tdata].ptr);

	for(int tid=0; tid<subtask_elems; tid++)
	{

	if(devParticles[tid].cell >= 0 && devParticles[tid].cell < NUM_CELLS) {

		int c, i, old;

		devTdata[tid].id = devParticles[tid].id;
		devTdata[tid].x = devParticles[tid].x;
		devTdata[tid].y = devParticles[tid].y;
		devTdata[tid].z = devParticles[tid].z;
		devTdata[tid].w = devParticles[tid].w;
		devTdata[tid].age = devParticles[tid].age;

		c = (1+MAX_PARTICLES_PER_CHUNK);
		i = devParticles[tid].chunk;
		old = serialAdd(&devChunkgrid[i*c+0], 1);
		if(old < MAX_PARTICLES_PER_CHUNK) {
			devChunkgrid[i*c+(old+1)] = devParticles[tid].id;
			serialMax(&devGridMax[0], old+1);
		}

		c = (1+MAX_PARTICLES_PER_CELL);
		i = devParticles[tid].cell;
		old = serialAdd(&devCellgrid[i*c+0], 1);
		if(old < MAX_PARTICLES_PER_CELL) {
			devCellgrid[i*c+(old+1)] = devParticles[tid].id;
			serialMax(&devGridMax[1], old+1);
		}
		else {
			serialAdd(&devCellgrid[i*c+0], -1);
			//printf("ERROR! MAX_PARTICLES_PER_CELL(%d) Overflow.\n", MAX_PARTICLES_PER_CELL);
			//printf(".");

			//kill
			reset_particle(devParticles[tid]);

			//lock-free (serial)
			q_insert(devQueueInfo, devQueue, devParticles[tid].seg_type, devParticles[tid].seg_tid, devParticles[tid].id);

			//exit(1);
			//asm("trap;"); // kill all threads
			//asm("exit;"); // kill this thread only
		}
	}

	}

	return pmSuccess;
}

pmStatus particleSystem_init_iframe_cpu(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	HOST_HEADER

	int index_mem_chunkgrid = 0;
	int index_mem_cellgrid 	= 1;
	int index_mem_gridmax 	= 2;

	int* devChunkgrid 	= (int*)(pSubtaskInfo.memInfo[index_mem_chunkgrid].ptr);
	int* devCellgrid 	= (int*)(pSubtaskInfo.memInfo[index_mem_cellgrid].ptr);
	int* devGridMax 	= (int*)(pSubtaskInfo.memInfo[index_mem_gridmax].ptr);

	for(int tid=0; tid<subtask_elems; tid++)
	{

	int zero = 0;
	ACC3 acc3 = {-1, 0, 0, 0};

	if(tid < NUM_CHUNKS*(1+MAX_PARTICLES_PER_CHUNK)) {
		devChunkgrid[tid] = zero;
	}

	if(tid < NUM_CELLS*(1+MAX_PARTICLES_PER_CELL)) {
		devCellgrid[tid] = zero;
	}

	if(tid < 2) {
		devGridMax[tid] = zero;
	}

	}

	return pmSuccess;
}

pmStatus particleSystem_init_iframe_host(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	HOST_HEADER

	int index_mem_chunkgrid = 0;
	int index_mem_cellgrid 	= 1;
	int index_mem_gridmax 	= 2;

	int* devChunkgrid 	= (int*)(pSubtaskInfo.memInfo[index_mem_chunkgrid].ptr);
	int* devCellgrid 	= (int*)(pSubtaskInfo.memInfo[index_mem_cellgrid].ptr);
	int* devGridMax 	= (int*)(pSubtaskInfo.memInfo[index_mem_gridmax].ptr);

	for(int tid=0; tid<subtask_elems; tid++)
	{

	int zero = 0;

	if(tid < NUM_CHUNKS*(1+MAX_PARTICLES_PER_CHUNK)) {
		devChunkgrid[tid] = zero;
	}

	if(tid < NUM_CELLS*(1+MAX_PARTICLES_PER_CELL)) {
		devCellgrid[tid] = zero;
	}

	if(tid < 2) {
		devGridMax[tid] = zero;
	}

	}

	return pmSuccess;
}

void serialparticleSystem(P_DATA_TYPE* pInputX, P_DATA_TYPE* pOutputA, int pos, int n, int dim)
{
}

// Returns execution time on success; 0 on error
double DoSerialProcess(int argc, char** argv, int pCommonArgs)
{
	READ_NON_COMMON_ARGS

	double lStartTime = getCurrentTimeInSecs();

	//serialparticleSystem(gSystemInput, gSerialOutput, 0, lMaxParticles, lMaxParticles);

	double lEndTime = getCurrentTimeInSecs();

	return (lEndTime - lStartTime);
}

/* NOT getting called */
// Returns execution time on success; 0 on error
double DoSingleGpuProcess(int argc, char** argv, int pCommonArgs)
{
printf("DoSingleGpuProcess\n");
#ifdef BUILD_CUDA
	READ_NON_COMMON_ARGS

	double lStartTime = getCurrentTimeInSecs();

	//singleGpuparticleSystem(gSystemInput, gParallelOutput, lMaxParticles);

	double lEndTime = getCurrentTimeInSecs();

	return (lEndTime - lStartTime);
#else
    return 0;
#endif
}

bool StageParallelTask_n(int numSubtasks, size_t numElemsPerSubtask, pmCallbackHandle *pCallbackHandles, pmMemHandle *handle_mem_list, int taskType, pmSchedulingPolicy pSchedulingPolicy, int iteration)
{
	CREATE_TASK(numSubtasks, pCallbackHandles[taskType], pSchedulingPolicy);

	int mem_count = 0;
	pmTaskMem lTaskMem[MAX_INDEX_MEM];

	if(taskType == 0) {
		lTaskMem[mem_count++] = {handle_mem_list[INDEX_MEM_PARTICLES], WRITE_ONLY, SUBSCRIPTION_OPTIMAL};  	/*WRITE*/
		lTaskMem[mem_count++] = {handle_mem_list[INDEX_MEM_TDATA], WRITE_ONLY, SUBSCRIPTION_OPTIMAL};		/*WRITE*/
	}
	else if(taskType == 1) {
		lTaskMem[mem_count++] = {handle_mem_list[INDEX_MEM_RANDSTATES], WRITE_ONLY, SUBSCRIPTION_OPTIMAL};  	/*WRITE*/
	}
	else if(taskType == 2) {
		lTaskMem[mem_count++] = {handle_mem_list[INDEX_MEM_QUEUEINFO], WRITE_ONLY, SUBSCRIPTION_OPTIMAL};	/*WRITE*/
		lTaskMem[mem_count++] = {handle_mem_list[INDEX_MEM_QUEUE], WRITE_ONLY, SUBSCRIPTION_OPTIMAL};  		/*WRITE*/
	}
	else if(taskType == 3) {
		lTaskMem[mem_count++] = {handle_mem_list[INDEX_MEM_CHUNKGRID], WRITE_ONLY, SUBSCRIPTION_OPTIMAL}; 	/*WRITE*/
		lTaskMem[mem_count++] = {handle_mem_list[INDEX_MEM_CELLGRID], WRITE_ONLY, SUBSCRIPTION_OPTIMAL};  	/*WRITE*/
		lTaskMem[mem_count++] = {handle_mem_list[INDEX_MEM_GRIDMAX], WRITE_ONLY, SUBSCRIPTION_OPTIMAL};  	/*WRITE*/
	}
	else if(taskType == 4) {
		lTaskMem[mem_count++] = {handle_mem_list[INDEX_MEM_PKGDISTRIB], WRITE_ONLY, SUBSCRIPTION_OPTIMAL}; 	/*WRITE*/
	}
	else if(taskType == 5) {
		lTaskMem[mem_count++] = {handle_mem_list[INDEX_MEM_QUEUEINFO], READ_WRITE, SUBSCRIPTION_OPTIMAL, true};	/*READ-WRITE*/
		lTaskMem[mem_count++] = {handle_mem_list[INDEX_MEM_QUEUE], READ_WRITE, SUBSCRIPTION_OPTIMAL, true};  	/*READ-WRITE*/
		lTaskMem[mem_count++] = {handle_mem_list[INDEX_MEM_PARTICLES], WRITE_ONLY, SUBSCRIPTION_OPTIMAL};  	/*WRITE*/
	}
	else if(taskType == 6) {
		lTaskMem[mem_count++] = {handle_mem_list[INDEX_MEM_CHUNKGRID], READ_ONLY, SUBSCRIPTION_OPTIMAL};  	/*READ*/
		lTaskMem[mem_count++] = {handle_mem_list[INDEX_MEM_CELLGRID], READ_ONLY, SUBSCRIPTION_OPTIMAL};  	/*READ*/
		lTaskMem[mem_count++] = {handle_mem_list[INDEX_MEM_PKGDISTRIB], READ_ONLY, SUBSCRIPTION_OPTIMAL}; 	/*READ*/
		lTaskMem[mem_count++] = {handle_mem_list[INDEX_MEM_QUEUEINFO], READ_WRITE, SUBSCRIPTION_OPTIMAL, true};	/*READ-WRITE*/

		lTaskMem[mem_count++] = {handle_mem_list[INDEX_MEM_QUEUE], READ_WRITE, SUBSCRIPTION_OPTIMAL, true};  	/*READ-WRITE*/
		lTaskMem[mem_count++] = {handle_mem_list[INDEX_MEM_PARTICLES], READ_WRITE, SUBSCRIPTION_OPTIMAL, true}; /*READ-WRITE*/
		lTaskMem[mem_count++] = {handle_mem_list[INDEX_MEM_TDATA], READ_ONLY, SUBSCRIPTION_OPTIMAL};		/*READ*/
	}
	else if(taskType == 8) {
		lTaskMem[mem_count++] = {handle_mem_list[INDEX_MEM_CHUNKGRID], READ_WRITE, SUBSCRIPTION_OPTIMAL, true};	/*READ-WRITE*/
		lTaskMem[mem_count++] = {handle_mem_list[INDEX_MEM_CELLGRID], READ_WRITE, SUBSCRIPTION_OPTIMAL, true};  /*READ-WRITE*/
		lTaskMem[mem_count++] = {handle_mem_list[INDEX_MEM_GRIDMAX], READ_WRITE, SUBSCRIPTION_OPTIMAL, true};  	/*READ-WRITE*/
		lTaskMem[mem_count++] = {handle_mem_list[INDEX_MEM_QUEUEINFO], READ_WRITE, SUBSCRIPTION_OPTIMAL, true};	/*READ-WRITE*/
		lTaskMem[mem_count++] = {handle_mem_list[INDEX_MEM_QUEUE], READ_WRITE, SUBSCRIPTION_OPTIMAL, true};  	/*READ-WRITE*/
		lTaskMem[mem_count++] = {handle_mem_list[INDEX_MEM_PARTICLES], READ_WRITE, SUBSCRIPTION_OPTIMAL, true};	/*READ-WRITE*/
		lTaskMem[mem_count++] = {handle_mem_list[INDEX_MEM_TDATA], WRITE_ONLY, SUBSCRIPTION_OPTIMAL};		/*WRITE*/
	}

	particleSystemTaskConf lTaskConf;
	lTaskConf.subtaskElems = numElemsPerSubtask;
	lTaskConf.iteration = iteration;
	#ifdef BUILD_CUDA
	    lTaskConf.cudaLaunchConf = GetCudaLaunchConf(numElemsPerSubtask);
	#endif

	lTaskDetails.taskMem = (pmTaskMem*)lTaskMem;
	lTaskDetails.taskMemCount = mem_count;
	lTaskDetails.taskConf = (void*)(&lTaskConf);
	lTaskDetails.taskConfLength = sizeof(lTaskConf);
//	lTaskDetails.taskId = pTaskId; 	// xxxxxxxxxxxxxxxx prioritize pTaskId ???????????????

	SAFE_PM_EXEC( pmSubmitTask(lTaskDetails, &lTaskHandle) );

	if(pmWaitForTaskCompletion(lTaskHandle) != pmSuccess)
	{
		FREE_TASK_AND_RESOURCES
		return false;
	}

	pmReleaseTask(lTaskHandle);

	return true;
}

bool StageParallelTask(int numSubtasks, size_t numElemsPerSubtask, pmCallbackHandle *pCallbackHandles, pmMemHandle *handle_mem_list, int taskType, pmSchedulingPolicy pSchedulingPolicy)
{
	return StageParallelTask_n(numSubtasks, numElemsPerSubtask, pCallbackHandles, handle_mem_list, taskType, pSchedulingPolicy, 0);
}

// Returns execution time on success; 0 on error
double DoParallelProcess(int argc, char** argv, int pCommonArgs, pmCallbackHandle* pCallbackHandles, pmSchedulingPolicy pSchedulingPolicy, bool pFetchBack)
{
	READ_NON_COMMON_ARGS

	/**********************************************************/

	printf("\n>>> MAX_PARTICLES_NUM = %d\n", MAX_PARTICLES_NUM);
	printf(">>> X_FACTOR = %d\n\n", X_FACTOR);

	printf(">>> CHUNK_FACTOR = %d\n", CHUNK_FACTOR);
	printf(">>> CHUNK_DIM = %d\n", CHUNK_DIM);
	printf(">>> GRID_DIM = %d\n", GRID_DIM);
	printf(">>> NUM_CHUNKS = %d\n", NUM_CHUNKS);
	printf(">>> NUM_CELLS = %d\n\n", NUM_CELLS);

	printf(">>> MAX_PARTICLES_PER_CELL = %d\n", MAX_PARTICLES_PER_CELL);
	printf(">>> CONTAINER_SIZE = %d (%d + %d + %d + %d)\n\n", CONTAINER_SIZE, SEG1_SIZE, SEG2_SIZE, SEG4_SIZE, SEG8_SIZE);

	/**********************************************************/

	double lStartTime = getCurrentTimeInSecs();

	pmMemHandle handle_mem_list[MAX_INDEX_MEM];
	handle_mem_list[INDEX_MEM_PARTICLES] 	= pm_mem_alloc(sizeParticles);
	handle_mem_list[INDEX_MEM_TDATA] 	= pm_mem_alloc(sizeTdata);
	handle_mem_list[INDEX_MEM_QUEUE] 	= pm_mem_alloc(sizeQueue);
	handle_mem_list[INDEX_MEM_QUEUEINFO] 	= pm_mem_alloc(sizeQueueInfo);
	handle_mem_list[INDEX_MEM_CHUNKGRID] 	= pm_mem_alloc(sizeChunkgrid);
	handle_mem_list[INDEX_MEM_CELLGRID] 	= pm_mem_alloc(sizeCellgrid);
	handle_mem_list[INDEX_MEM_GRIDMAX] 	= pm_mem_alloc(sizeGridMax);
	handle_mem_list[INDEX_MEM_PKGDISTRIB] 	= pm_mem_alloc(sizePkgDistrib);
	handle_mem_list[INDEX_MEM_RANDSTATES] 	= pm_mem_alloc(sizeRandstates);

	//-----------------------------

	// initialize particles
	if(SERIAL_MODE == 1) {
		StageParallelTask(1, CONTAINER_SIZE, pCallbackHandles, handle_mem_list, 0, pSchedulingPolicy);
	}
	else {
		int num_frags = NUM_FRAGS;
		int frag_len = (CONTAINER_SIZE+num_frags-1)/num_frags;
		StageParallelTask(num_frags, frag_len, pCallbackHandles, handle_mem_list, 0, pSchedulingPolicy);
	}

	if(pFetchBack)
	{
//////////// comment
		SAFE_PM_EXEC( pmFetchMemory(handle_mem_list[INDEX_MEM_PARTICLES]) );
		SAFE_PM_EXEC( pmFetchMemory(handle_mem_list[INDEX_MEM_TDATA]) );
		pm_mem_copyout(hostParticles, handle_mem_list[INDEX_MEM_PARTICLES], 0, sizeParticles);
		pm_mem_copyout(hostTdata, handle_mem_list[INDEX_MEM_TDATA], 0, sizeTdata);
////////////
	}

	//-----------------------------

	if( SERIAL_MODE == 0 && HOST_MODE == 0 ) {
	// set up seeds
	StageParallelTask(1, CONTAINER_SIZE, pCallbackHandles, handle_mem_list, 1, pSchedulingPolicy);

	if(pFetchBack)
	{
		SAFE_PM_EXEC( pmFetchMemory(handle_mem_list[INDEX_MEM_RANDSTATES]) );
		pm_mem_copyout(hostRandstates, handle_mem_list[INDEX_MEM_RANDSTATES], 0, sizeRandstates);
	}
	}

	//-----------------------------

	// initialize and (fast) fill queue
	StageParallelTask(1, CONTAINER_SIZE, pCallbackHandles, handle_mem_list, 2, pSchedulingPolicy);

	if(pFetchBack)
	{
		SAFE_PM_EXEC( pmFetchMemory(handle_mem_list[INDEX_MEM_QUEUEINFO]) );
		SAFE_PM_EXEC( pmFetchMemory(handle_mem_list[INDEX_MEM_QUEUE]) );
		pm_mem_copyout(hostQueueInfo, handle_mem_list[INDEX_MEM_QUEUEINFO], 0, sizeQueueInfo);
		pm_mem_copyout(hostQueue, handle_mem_list[INDEX_MEM_QUEUE], 0, sizeQueue);
	}

	//-----------------------------

	// fill (setup) particles
	StageParallelTask(1, CONTAINER_SIZE, pCallbackHandles, handle_mem_list, 5, pSchedulingPolicy);
	if(pFetchBack)
	{
//////////// comment
		SAFE_PM_EXEC( pmFetchMemory(handle_mem_list[INDEX_MEM_QUEUEINFO]) );
		SAFE_PM_EXEC( pmFetchMemory(handle_mem_list[INDEX_MEM_QUEUE]) );
		SAFE_PM_EXEC( pmFetchMemory(handle_mem_list[INDEX_MEM_PARTICLES]) );
		pm_mem_copyout(hostQueueInfo, handle_mem_list[INDEX_MEM_QUEUEINFO], 0, sizeQueueInfo);
		pm_mem_copyout(hostQueue, handle_mem_list[INDEX_MEM_QUEUE], 0, sizeQueue);
		pm_mem_copyout(hostParticles, handle_mem_list[INDEX_MEM_PARTICLES], 0, sizeParticles);
////////////
	}

	//-----------------------------

	// compute package distribution
	StageParallelTask(1, NUM_CHUNKS, pCallbackHandles, handle_mem_list, 4, pSchedulingPolicy);

	if(pFetchBack)
	{
		SAFE_PM_EXEC( pmFetchMemory(handle_mem_list[INDEX_MEM_PKGDISTRIB]) );
		pm_mem_copyout(hostPkgDistrib, handle_mem_list[INDEX_MEM_PKGDISTRIB], 0, sizePkgDistrib);
	}

	//-----------------------------

	for(int niter=0; niter<1; niter++) 
	{

	double time0 = getCurrentTimeInSecs();
	int biggestChunkSize;

	//-----------------------------

	// initialize Chunkgrid, Cellgrid, GridMax
	int sizes[] = {nChunkgrid, nCellgrid};  // nCellgrid
	int max_size = *std::max_element(sizes,sizes+2);

	StageParallelTask(1, max_size, pCallbackHandles, handle_mem_list, 3, pSchedulingPolicy);

	if(pFetchBack)
	{
		SAFE_PM_EXEC( pmFetchMemory(handle_mem_list[INDEX_MEM_CHUNKGRID]) );
		SAFE_PM_EXEC( pmFetchMemory(handle_mem_list[INDEX_MEM_CELLGRID]) );
		SAFE_PM_EXEC( pmFetchMemory(handle_mem_list[INDEX_MEM_GRIDMAX]) );
		pm_mem_copyout(hostChunkgrid, handle_mem_list[INDEX_MEM_CHUNKGRID], 0, sizeChunkgrid);
		pm_mem_copyout(hostCellgrid, handle_mem_list[INDEX_MEM_CELLGRID], 0, sizeCellgrid);
/***/		pm_mem_copyout(hostGridMax, handle_mem_list[INDEX_MEM_GRIDMAX], 0, sizeGridMax);
	}

	//-----------------------------

	double time1 = getCurrentTimeInSecs();

	// build grid, tdata snapshot
	StageParallelTask(1, CONTAINER_SIZE, pCallbackHandles, handle_mem_list, 8, pSchedulingPolicy);

	if(pFetchBack)
	{
		SAFE_PM_EXEC( pmFetchMemory(handle_mem_list[INDEX_MEM_CHUNKGRID]) );
		SAFE_PM_EXEC( pmFetchMemory(handle_mem_list[INDEX_MEM_CELLGRID]) );
		SAFE_PM_EXEC( pmFetchMemory(handle_mem_list[INDEX_MEM_GRIDMAX]) );
		SAFE_PM_EXEC( pmFetchMemory(handle_mem_list[INDEX_MEM_QUEUEINFO]) );
		SAFE_PM_EXEC( pmFetchMemory(handle_mem_list[INDEX_MEM_QUEUE]) );
		SAFE_PM_EXEC( pmFetchMemory(handle_mem_list[INDEX_MEM_PARTICLES]) );
		SAFE_PM_EXEC( pmFetchMemory(handle_mem_list[INDEX_MEM_TDATA]) );
		pm_mem_copyout(hostChunkgrid, handle_mem_list[INDEX_MEM_CHUNKGRID], 0, sizeChunkgrid);
		pm_mem_copyout(hostCellgrid, handle_mem_list[INDEX_MEM_CELLGRID], 0, sizeCellgrid);
/***/		pm_mem_copyout(hostGridMax, handle_mem_list[INDEX_MEM_GRIDMAX], 0, sizeGridMax);
		pm_mem_copyout(hostQueueInfo, handle_mem_list[INDEX_MEM_QUEUEINFO], 0, sizeQueueInfo);
		pm_mem_copyout(hostQueue, handle_mem_list[INDEX_MEM_QUEUE], 0, sizeQueue);
		pm_mem_copyout(hostParticles, handle_mem_list[INDEX_MEM_PARTICLES], 0, sizeParticles);
		pm_mem_copyout(hostTdata, handle_mem_list[INDEX_MEM_TDATA], 0, sizeTdata);
	}

	//-----------------------------

	double time2 = getCurrentTimeInSecs();
	// particle collision, death
	// calculate forces/acceleration
	// update position and velocity
	// handle (seg fault, particle explosion)
	// _x
	biggestChunkSize = hostGridMax[0];
	if(biggestChunkSize > 0) {

		int num = NUM_CHUNKS;
		int len = biggestChunkSize;

		//StageParallelTask(num, len, pCallbackHandles, handle_mem_list, 6, pSchedulingPolicy);
		for(int i=0; i<((num+ST_MAX-1)/ST_MAX); i++) {
			int n = (ST_MAX*(i+1)<=num) ? ST_MAX : (num)-(ST_MAX*i);
			StageParallelTask_n(n, len, pCallbackHandles, handle_mem_list, 6, pSchedulingPolicy, i);
		}

	}

	if(pFetchBack)
	{
		SAFE_PM_EXEC( pmFetchMemory(handle_mem_list[INDEX_MEM_QUEUEINFO]) );
		SAFE_PM_EXEC( pmFetchMemory(handle_mem_list[INDEX_MEM_QUEUE]) );
		SAFE_PM_EXEC( pmFetchMemory(handle_mem_list[INDEX_MEM_PARTICLES]) );
		pm_mem_copyout(hostQueueInfo, handle_mem_list[INDEX_MEM_QUEUEINFO], 0, sizeQueueInfo);
		pm_mem_copyout(hostQueue, handle_mem_list[INDEX_MEM_QUEUE], 0, sizeQueue);
		pm_mem_copyout(hostParticles, handle_mem_list[INDEX_MEM_PARTICLES], 0, sizeParticles);
	}

	//-----------------------------

	double time3 = getCurrentTimeInSecs();
	printf(">>>>>>>>>>> Time per iteration (sec): \n%f\n%f\n%f\n%f\n\n\n\n", time3-time0, time1-time0, time2-time1, time3-time2);
	}

	double lEndTime = getCurrentTimeInSecs();


/**********************************************************
	printf("--------------------- hostParticles\n");
	for(int i=0; i<CONTAINER_SIZE; i++) {  // CONTAINER_SIZE
//		if(i<5 || hostParticles[i].cell >= 0) printf("[%d] %d %d: (%6.2f,%6.2f,%6.2f) (%6.2f,%6.2f,%6.2f)\n", hostParticles[i].id, hostParticles[i].chunk, hostParticles[i].cell, hostParticles[i].x, hostParticles[i].y, hostParticles[i].z, hostParticles[i].vx, hostParticles[i].vy, hostParticles[i].vz);
//		if(i<5 || hostParticles[i].cell >= 0) printf("[%d] %d %d (%d %d) %f %f: (%6.2f,%6.2f,%6.2f) (%6.2f,%6.2f,%6.2f)\n"
		if((i>=2*(CONTAINER_SIZE/64)-5 && i<2*(CONTAINER_SIZE/64)+5) || hostParticles[i].cell >= 0) printf("[%d] %d %d (%d %d) %f %f: (%6.2f,%6.2f,%6.2f) (%6.2f,%6.2f,%6.2f) (%6.2f,%6.2f,%6.2f)\n"
//		if(i<5) printf("[%d] %d %d (%d %d) %f %f: (%6.2f,%6.2f,%6.2f) (%6.2f,%6.2f,%6.2f)\n"
			, hostParticles[i].id, hostParticles[i].chunk, hostParticles[i].cell, hostParticles[i].seg_type, hostParticles[i].seg_tid, hostParticles[i].age, hostParticles[i].fertility_age 
			, hostParticles[i].x, hostParticles[i].y, hostParticles[i].z
			, hostParticles[i].vx, hostParticles[i].vy, hostParticles[i].vz
			, hostParticles[i].ax, hostParticles[i].ay, hostParticles[i].az);
	}
	printf("--------------------- hostRandstates\n");
	for(int i=0; i<10; i++) {  // CONTAINER_SIZE
		printf(" [%d] >>>::  %f\n", i, curand_uniform(hostRandstates + i));
	}
	printf("--------------------- hostQueueInfo, hostQueue\n");
	for(int i=0; i<QUEUE_INFO_SIZE; i++) {  // QUEUE_INFO_SIZE
//		if(i==0 || hostQueueInfo[i].count > 0) {
//--		if(i >= (SEG1_COUNT + SEG2_COUNT + SEG4_COUNT)) {
		if(hostQueueInfo[i].seg_size != hostQueueInfo[i].count) {
			printf(" >>>> [%d] %d %d %d %d | %d %d |  ", i, hostQueueInfo[i].front, hostQueueInfo[i].rear, hostQueueInfo[i].count, hostQueueInfo[i].lock, hostQueueInfo[i].rloc, hostQueueInfo[i].seg_size);
			for(int j=hostQueueInfo[i].rloc; j<hostQueueInfo[i].rloc+hostQueueInfo[i].seg_size; j++) {
				if(j<hostQueueInfo[i].rloc+10) printf(" :%d", hostQueue[j]);
			}
			printf("\n");
		}
//--		}
//		}
	}
	printf("--------------------- hostPkgDistrib\n");
	for(int i=0; i<27*NUM_CHUNKS; i++) {  // 27*NUM_CHUNKS
		if(i/27 == 3) printf(" %2d [%2d] ::>>  %3d %3d\n", i/27, i%27, hostPkgDistrib[i].c, hostPkgDistrib[i].p);
	}
	printf("--------------------- hostGridMax\n");
	printf(" # : %d %d\n", hostGridMax[0], hostGridMax[1]);
	printf("--------------------- hostChunkgrid\n");
	for(int i=0; i<NUM_CHUNKS; i++) {  // NUM_CHUNKS
		int c = (1+MAX_PARTICLES_PER_CHUNK);
		if(i<5 || (hostChunkgrid[i*c+0] > 0)) printf(" [%d] %d : %d %d %d %d\n", i, hostChunkgrid[i*c+0], hostChunkgrid[i*c+1], hostChunkgrid[i*c+2], hostChunkgrid[i*c+3], hostChunkgrid[i*c+4]);
//		if(i<5) printf(" [%d] %d : %d %d %d %d\n", i, hostChunkgrid[i*c+0], hostChunkgrid[i*c+1], hostChunkgrid[i*c+2], hostChunkgrid[i*c+3], hostChunkgrid[i*c+4]);
	}
	printf("--------------------- hostCellgrid\n");
	for(int i=0; i<NUM_CELLS; i++) {  // NUM_CELLS
		int c = (1+MAX_PARTICLES_PER_CELL);
		if(i<5 || (hostCellgrid[i*c+0] > 0)) printf(" [%d] %d : %d %d %d\n", i, hostCellgrid[i*c+0], hostCellgrid[i*c+1], hostCellgrid[i*c+2], hostCellgrid[i*c+3]);
//		if(i<5) printf(" [%d] %d : %d %d %d\n", i, hostCellgrid[i*c+0], hostCellgrid[i*c+1], hostCellgrid[i*c+2], hostCellgrid[i*c+3]);
	}
	printf("---------------------\n");

/**********************************************************/

	return (lEndTime - lStartTime);
}

pmCallbacks DoSetDefaultCallbacks_init_particles()
{
	pmCallbacks lCallbacks;
	
	if(SERIAL_MODE == 1) {
		lCallbacks.dataDistribution = particleSystemDataDistribution_00_s;
		lCallbacks.deviceSelection = NULL;
		lCallbacks.subtask_cpu = particleSystem_init_particles_cpu;
	}
	else if(HOST_MODE == 1) {
		lCallbacks.dataDistribution = particleSystemDataDistribution_00;
		lCallbacks.deviceSelection = NULL;
		lCallbacks.subtask_cpu = particleSystem_init_particles_host;
	}
	else {
		lCallbacks.dataDistribution = particleSystemDataDistribution_00;
		lCallbacks.deviceSelection = NULL;

		#ifdef BUILD_CUDA
		lCallbacks.subtask_gpu_cuda = particleSystem_init_particles_cudaFunc;
		#endif
	}

	return lCallbacks;
}

pmCallbacks DoSetDefaultCallbacks_fill_particles()
{
	pmCallbacks lCallbacks;

	lCallbacks.dataDistribution = particleSystemDataDistribution_05_s;
	lCallbacks.deviceSelection = NULL;
	lCallbacks.subtask_cpu = particleSystem_fill_particles_cpu;

	#ifdef BUILD_CUDA
//	lCallbacks.subtask_gpu_cuda = particleSystem_cudaFunc;
	#endif

	return lCallbacks;
}

pmCallbacks DoSetDefaultCallbacks_setup_seeds()
{
	pmCallbacks lCallbacks;

	lCallbacks.dataDistribution = particleSystemDataDistribution_01;
	lCallbacks.deviceSelection = NULL;
//	lCallbacks.subtask_cpu = particleSystem_cpu;

	#ifdef BUILD_CUDA
	lCallbacks.subtask_gpu_cuda = particleSystem_setup_seeds_cudaFunc;
	#endif

	return lCallbacks;
}

pmCallbacks DoSetDefaultCallbacks_q_start_fast()
{
	pmCallbacks lCallbacks;

	if(SERIAL_MODE == 1) {
		lCallbacks.dataDistribution = particleSystemDataDistribution_02_s;
		lCallbacks.deviceSelection = NULL;
		lCallbacks.subtask_cpu = particleSystem_q_start_fast_cpu;
	}
	else if(HOST_MODE == 1) {
		lCallbacks.dataDistribution = particleSystemDataDistribution_02;
		lCallbacks.deviceSelection = NULL;
		lCallbacks.subtask_cpu = particleSystem_q_start_fast_host;
	}
	else {
		lCallbacks.dataDistribution = particleSystemDataDistribution_02;
		lCallbacks.deviceSelection = NULL;

		#ifdef BUILD_CUDA
		lCallbacks.subtask_gpu_cuda = particleSystem_q_start_fast_cudaFunc;
		#endif
	}

	return lCallbacks;
}

pmCallbacks DoSetDefaultCallbacks_init_iframe()
{
	pmCallbacks lCallbacks;

	if(SERIAL_MODE == 1) {
		lCallbacks.dataDistribution = particleSystemDataDistribution_03_s;
		lCallbacks.deviceSelection = NULL;
		lCallbacks.subtask_cpu = particleSystem_init_iframe_cpu;
	}
	else if(HOST_MODE == 1) {
		lCallbacks.dataDistribution = particleSystemDataDistribution_03;
		lCallbacks.deviceSelection = NULL;
		lCallbacks.subtask_cpu = particleSystem_init_iframe_host;
	}
	else {
		lCallbacks.dataDistribution = particleSystemDataDistribution_03;
		lCallbacks.deviceSelection = NULL;

		#ifdef BUILD_CUDA
		lCallbacks.subtask_gpu_cuda = particleSystem_init_iframe_cudaFunc;
		#endif
	}

	return lCallbacks;
}

pmCallbacks DoSetDefaultCallbacks_calc_forces()
{
	pmCallbacks lCallbacks;

	if(SERIAL_MODE == 1) {
		lCallbacks.dataDistribution = particleSystemDataDistribution_06_s;
		lCallbacks.deviceSelection = NULL;

		lCallbacks.subtask_cpu = particleSystem_calc_forces_cpu;
	}
	else if(HOST_MODE == 1) {
		lCallbacks.dataDistribution = particleSystemDataDistribution_06;
		lCallbacks.deviceSelection = NULL;
		lCallbacks.subtask_cpu = particleSystem_calc_forces_host;
	}
	else {
		lCallbacks.dataDistribution = particleSystemDataDistribution_06;
		lCallbacks.deviceSelection = NULL;

		#ifdef BUILD_CUDA
		lCallbacks.subtask_gpu_cuda = particleSystem_calc_forces_cudaFunc;
		#endif
	}

	return lCallbacks;
}

pmCallbacks DoSetDefaultCallbacks_collision()
{
	pmCallbacks lCallbacks;
	return lCallbacks;
}

pmCallbacks DoSetDefaultCallbacks_update_xv()
{
	pmCallbacks lCallbacks;
	return lCallbacks;
}

pmCallbacks DoSetDefaultCallbacks_update_sys1()
{
	pmCallbacks lCallbacks;
	return lCallbacks;
}

pmCallbacks DoSetDefaultCallbacks_update_sys2()
{
	pmCallbacks lCallbacks;
	return lCallbacks;
}

pmCallbacks DoSetDefaultCallbacks_pkg_distrib()
{
	pmCallbacks lCallbacks;

	if(SERIAL_MODE == 1) {
		lCallbacks.dataDistribution = particleSystemDataDistribution_04_s;
		lCallbacks.deviceSelection = NULL;
		lCallbacks.subtask_cpu = particleSystem_pkg_distrib_cpu;
	}
	else if(HOST_MODE == 1) {
		lCallbacks.dataDistribution = particleSystemDataDistribution_04;
		lCallbacks.deviceSelection = NULL;
		lCallbacks.subtask_cpu = particleSystem_pkg_distrib_host;
	}
	else {
		lCallbacks.dataDistribution = particleSystemDataDistribution_04;
		lCallbacks.deviceSelection = NULL;

		#ifdef BUILD_CUDA
		lCallbacks.subtask_gpu_cuda = particleSystem_pkg_distrib_cudaFunc;
		#endif
	}

	return lCallbacks;
}

pmCallbacks DoSetDefaultCallbacks_build_grid()
{
	pmCallbacks lCallbacks;

	if(SERIAL_MODE == 1) {
		lCallbacks.dataDistribution = particleSystemDataDistribution_08_s;
		lCallbacks.deviceSelection = NULL;
		lCallbacks.subtask_cpu = particleSystem_build_grid_cpu;
	}
	else if(HOST_MODE == 1) {
		lCallbacks.dataDistribution = particleSystemDataDistribution_08;
		lCallbacks.deviceSelection = NULL;
		lCallbacks.subtask_cpu = particleSystem_build_grid_host;
	}
	else {
		lCallbacks.dataDistribution = particleSystemDataDistribution_08;
		lCallbacks.deviceSelection = NULL;

		#ifdef BUILD_CUDA
		lCallbacks.subtask_gpu_cuda = particleSystem_build_grid_cudaFunc;
		#endif
	}

	return lCallbacks;
}

// Returns 0 on success; non-zero on failure
int DoInit(int argc, char** argv, int pCommonArgs)
{
	READ_NON_COMMON_ARGS

	nParticles	= CONTAINER_SIZE;
	nTdata		= CONTAINER_SIZE;
	nQueue		= CONTAINER_SIZE;
	nQueueInfo	= QUEUE_INFO_SIZE;
	nChunkgrid	= NUM_CHUNKS*(1+MAX_PARTICLES_PER_CHUNK);
	nCellgrid	= NUM_CELLS*(1+MAX_PARTICLES_PER_CELL);
	nGridMax	= 2;
	nPkgDistrib  	= NUM_CHUNKS*27;
	nRandstates	= CONTAINER_SIZE;

	sizeParticles 	= nParticles	* sizeof(P_DATA_TYPE);
	sizeTdata 	= nTdata	* sizeof(T_DATA_TYPE);
	sizeQueue 	= nQueue 	* sizeof(int);
	sizeQueueInfo 	= nQueueInfo 	* sizeof(QUEUE_INFO);
	sizeChunkgrid 	= nChunkgrid 	* sizeof(int);
	sizeCellgrid 	= nCellgrid 	* sizeof(int);
	sizeGridMax 	= nGridMax 	* sizeof(int);
	sizePkgDistrib 	= nPkgDistrib 	* sizeof(PAIR);
	sizeRandstates 	= nRandstates 	* sizeof(curandState);

	hostParticles 	= new P_DATA_TYPE[nParticles];
	hostTdata 	= new T_DATA_TYPE[nTdata];
	hostQueue	= new int[nQueue];
	hostQueueInfo	= new QUEUE_INFO[nQueueInfo];
	hostChunkgrid	= new int[nChunkgrid];
	hostCellgrid	= new int[nCellgrid];
	hostGridMax	= new int[nGridMax];
	hostPkgDistrib  = new PAIR[nPkgDistrib];
	hostRandstates	= new curandState[nRandstates];

	return 0;
}

// Returns 0 on success; non-zero on failure
int DoDestroy()
{
	delete[] hostParticles;
	delete[] hostTdata;
	delete[] hostQueue;
	delete[] hostQueueInfo;
	delete[] hostChunkgrid;
	delete[] hostCellgrid;
	delete[] hostGridMax;
	delete[] hostPkgDistrib;
	delete[] hostRandstates;

	return 0;
}

// Returns 0 if serial and parallel executions have produced same result; non-zero otherwise
int DoCompare(int argc, char** argv, int pCommonArgs)
{
	return 0;	// Serial Comparison Test Passed ALWAYS
}

/**	Non-common args
 *	1. Max number of particles
 */
int main(int argc, char** argv)
{
	// both parallel and serial versions are implemented
	// 1: only parallel version
	// 5, 4: only serial version
	// outr: 0, 1, 2, 4, 5
	// iter: 3, 6, 8
	callbackStruct lStruct[] = { 
		 {DoSetDefaultCallbacks_init_particles, "INIT_PARTICLES"}	// 0 :Initialize and fill
		,{DoSetDefaultCallbacks_setup_seeds, 	"SETUP_SEEDS"}		// 1
		,{DoSetDefaultCallbacks_q_start_fast, 	"QUEUE_FILL"} 	 	// 2 :Initialize and fill
		,{DoSetDefaultCallbacks_init_iframe, 	"INIT_IFRAME"}		// 3
		,{DoSetDefaultCallbacks_pkg_distrib, 	"PKG_DISTRIB"}		// 4
		,{DoSetDefaultCallbacks_fill_particles, "FILL_PARTICLES"}	// 5
		,{DoSetDefaultCallbacks_calc_forces, 	"CALC_FORCES"}		// 6
		,{DoSetDefaultCallbacks_update_xv, 	"UPDATE_XV"}		//x7
		,{DoSetDefaultCallbacks_build_grid, 	"BUILD_GRID"}		// 8
		,{DoSetDefaultCallbacks_update_sys1, 	"UPDATE_SYS1"}		//x9
		,{DoSetDefaultCallbacks_collision, 	"COLLISION"}		//x10
		,{DoSetDefaultCallbacks_update_sys2, 	"UPDATE_SYS2"}		//x11
	};

	int numCallbacks = sizeof(lStruct)/sizeof(lStruct[0]);

	commonStart(argc, argv, DoInit, DoSerialProcess, DoSingleGpuProcess, DoParallelProcess, DoCompare, DoDestroy, lStruct, numCallbacks);
	commonFinish();

	return 0;
}

}

