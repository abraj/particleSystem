
#ifdef BUILD_CUDA

#include "commonAPI.h"
#include "pmPublicDefinitions.h"

#include <curand_kernel.h>
//#include <iostream>
#include <stdio.h>

#include "particleSystem.h"
#include "app_common.cu"
#include "app.cu"

namespace particleSystem
{

__global__ void particleSystem_init_particles_cuda(pmTaskInfo pTaskInfo, pmDeviceInfo* pDeviceInfo, pmSubtaskInfo pSubtaskInfo, pmStatus* pStatus)
{
	KERNEL_HEADER

	int index_mem_particles = 0;

	P_DATA_TYPE* devParticles = (P_DATA_TYPE*)(pSubtaskInfo.memInfo[index_mem_particles].ptr);

	int num_frags = NUM_FRAGS;
	int frag_len = (CONTAINER_SIZE+num_frags-1)/num_frags;
	int gid = subtask_id * frag_len + tid;
	if(gid > CONTAINER_SIZE-1) return;

	devParticles[tid].id = gid;
	init_particle(devParticles[tid]);

	*pStatus = pmSuccess;
}

__global__ void particleSystem_setup_seeds_cuda(pmTaskInfo pTaskInfo, pmDeviceInfo* pDeviceInfo, pmSubtaskInfo pSubtaskInfo, pmStatus* pStatus)
{
	KERNEL_HEADER

	int index_mem_randstates = 0;

	curandState* devRandstates = (curandState*)(pSubtaskInfo.memInfo[index_mem_randstates].ptr);

	curand_init( RAND_SEED, tid, tid, devRandstates + tid );

	*pStatus = pmSuccess;
}

__global__ void particleSystem_q_start_fast_cuda(pmTaskInfo pTaskInfo, pmDeviceInfo* pDeviceInfo, pmSubtaskInfo pSubtaskInfo, pmStatus* pStatus)
{
	KERNEL_HEADER

	int seg_type, seg_tid;
	int rloc, seg_size;

	int index_mem_queueinfo = 0;
	int index_mem_queue = 1;

	QUEUE_INFO* devQueueInfo = (QUEUE_INFO*)(pSubtaskInfo.memInfo[index_mem_queueinfo].ptr);
	int* devQueue = (int*)(pSubtaskInfo.memInfo[index_mem_queue].ptr);

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

	*pStatus = pmSuccess;
}

__global__ void particleSystem_init_iframe_cuda(pmTaskInfo pTaskInfo, pmDeviceInfo* pDeviceInfo, pmSubtaskInfo pSubtaskInfo, pmStatus* pStatus)
{
	KERNEL_HEADER

	int index_mem_crossmap 	= 0;
	int index_mem_acc 	= 1;
	int index_mem_flags 	= 2;
	int index_mem_chunkgrid = 3;
	int index_mem_cellgrid 	= 4;
	int index_mem_gridmax 	= 5;

	int* devCrossMap 	= (int*)(pSubtaskInfo.memInfo[index_mem_crossmap].ptr);
	ACC3* devAcc		= (ACC3*)(pSubtaskInfo.memInfo[index_mem_acc].ptr);
	int* devFlags 		= (int*)(pSubtaskInfo.memInfo[index_mem_flags].ptr);
	int* devChunkgrid 	= (int*)(pSubtaskInfo.memInfo[index_mem_chunkgrid].ptr);
	int* devCellgrid 	= (int*)(pSubtaskInfo.memInfo[index_mem_cellgrid].ptr);
	int* devGridMax 	= (int*)(pSubtaskInfo.memInfo[index_mem_gridmax].ptr);

	int zero = 0;
	ACC3 acc3 = {-1, 0, 0, 0};

	if(tid < CONTAINER_SIZE) {
		devCrossMap[tid] = -1;
	}

	if(tid < NUM_CHUNKS*MAX_PARTICLES_PER_CHUNK) {
		devAcc[tid] = acc3;
	}

	if(tid < CONTAINER_SIZE) {
		devFlags[tid] = zero;
	}

	if(tid < NUM_CHUNKS*(1+MAX_PARTICLES_PER_CHUNK)) {
		devChunkgrid[tid] = zero;
	}

	if(tid < NUM_CELLS*(1+MAX_PARTICLES_PER_CELL)) {
		devCellgrid[tid] = zero;
	}

	if(tid < 2) {
		devGridMax[tid] = zero;
	}

	*pStatus = pmSuccess;
}

__global__ void particleSystem_calc_forces_cuda(pmTaskInfo pTaskInfo, pmDeviceInfo* pDeviceInfo, pmSubtaskInfo pSubtaskInfo, pmStatus* pStatus)
{
	KERNEL_HEADER

	*pStatus = pmSuccess;
}

__global__ void particleSystem_pkg_distrib_cuda(pmTaskInfo pTaskInfo, pmDeviceInfo* pDeviceInfo, pmSubtaskInfo pSubtaskInfo, pmStatus* pStatus)
{
	KERNEL_HEADER

	int index_mem_pkgdistrib = 0;

	PAIR* devPkgDistrib = (PAIR*)(pSubtaskInfo.memInfo[index_mem_pkgdistrib].ptr);

	if(tid >= NUM_CHUNKS) return;

	int chunk = tid;
	PAIR *seg_list = devPkgDistrib + (tid*27);
	set_pkg_segments(chunk, seg_list);

	*pStatus = pmSuccess;
}

__global__ void particleSystem_build_grid_cuda(pmTaskInfo pTaskInfo, pmDeviceInfo* pDeviceInfo, pmSubtaskInfo pSubtaskInfo, pmStatus* pStatus)
{
	KERNEL_HEADER

	int index_mem_chunkgrid = 0;
	int index_mem_cellgrid 	= 1;
	int index_mem_gridmax 	= 2;
	int index_mem_crossmap 	= 3;
	int index_mem_queueinfo = 4;
	int index_mem_queue 	= 5;
	int index_mem_particles = 6;

	int* devChunkgrid = (int*)(pSubtaskInfo.memInfo[index_mem_chunkgrid].ptr);
	int* devCellgrid = (int*)(pSubtaskInfo.memInfo[index_mem_cellgrid].ptr);
	int* devGridMax = (int*)(pSubtaskInfo.memInfo[index_mem_gridmax].ptr);
	int* devCrossMap = (int*)(pSubtaskInfo.memInfo[index_mem_crossmap].ptr);
	QUEUE_INFO* devQueueInfo = (QUEUE_INFO*)(pSubtaskInfo.memInfo[index_mem_queueinfo].ptr);
	int* devQueue = (int*)(pSubtaskInfo.memInfo[index_mem_queue].ptr);
	P_DATA_TYPE* devParticles = (P_DATA_TYPE*)(pSubtaskInfo.memInfo[index_mem_particles].ptr);

	if(devParticles[tid].cell >= 0 && devParticles[tid].cell < NUM_CELLS) {

		int c, i, old;

		c = (1+MAX_PARTICLES_PER_CHUNK);
		i = devParticles[tid].chunk;
		old = atomicAdd(&devChunkgrid[i*c+0], 1);
		if(old < MAX_PARTICLES_PER_CHUNK) {
			devChunkgrid[i*c+(old+1)] = devParticles[tid].id;
			devCrossMap[tid] = old;
			atomicMax(&devGridMax[0], old+1);
		}

		c = (1+MAX_PARTICLES_PER_CELL);
		i = devParticles[tid].cell;
		old = atomicAdd(&devCellgrid[i*c+0], 1);
		if(old < MAX_PARTICLES_PER_CELL) {
			devCellgrid[i*c+(old+1)] = devParticles[tid].id;
			atomicMax(&devGridMax[1], old+1);
		}
		else {
			atomicAdd(&devCellgrid[i*c+0], -1);
			//printf("ERROR! MAX_PARTICLES_PER_CELL(%d) Overflow.\n", MAX_PARTICLES_PER_CELL);
			//printf(".");

			//kill
			reset_particle(devParticles[tid]);

			//lock-secure
			q_insert(tid, devQueueInfo, devQueue, devParticles[tid].seg_type, devParticles[tid].seg_tid, devParticles[tid].id);

			//asm("trap;"); // kill all threads
			//asm("exit;"); // kill this thread only
		}
	}

	*pStatus = pmSuccess;
}

/* NOT getting called */
__global__ void particleSystem_singleGpu(void *devX, void *devA, int N, int p)
{
}

/* NOT getting called */
// Returns 0 on success; non-zero on failure
int singleGpuparticleSystem(P_DATA_TYPE* pInputX, P_DATA_TYPE* pOutputA, int pDim)
{
    return 0;
}

particleSystem_cudaFuncPtr particleSystem_init_particles_cudaFunc = particleSystem_init_particles_cuda;
particleSystem_cudaFuncPtr particleSystem_setup_seeds_cudaFunc = particleSystem_setup_seeds_cuda;
particleSystem_cudaFuncPtr particleSystem_q_start_fast_cudaFunc = particleSystem_q_start_fast_cuda;
particleSystem_cudaFuncPtr particleSystem_init_iframe_cudaFunc = particleSystem_init_iframe_cuda;
particleSystem_cudaFuncPtr particleSystem_calc_forces_cudaFunc = particleSystem_calc_forces_cuda;
particleSystem_cudaFuncPtr particleSystem_pkg_distrib_cudaFunc = particleSystem_pkg_distrib_cuda;
particleSystem_cudaFuncPtr particleSystem_build_grid_cudaFunc = particleSystem_build_grid_cuda;

}

#endif

