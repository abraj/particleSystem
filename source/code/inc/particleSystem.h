#ifndef __PARTICLESYSTEM_H__
#define __PARTICLESYSTEM_H__

#include "common.h"

INT3 get_cell_info(int cell);
int get_cont_rloc(int seg_type, int seg_tid);
int get_info_rloc(int seg_type, int seg_tid);
void set_pkg_segments(int chunk, PAIR *seg_list);
void create_particle_s(P_DATA_TYPE *devX, int gtid, float w, float age, float fert_age, float x, float y, float z, float vx, float vy, float vz);
int get_local_pos(int particle_id, PAIR *seg_list);
int get_natural_pos(int particle_id, int subtask_id);
void fill_cells(NEIB_CELLS &neibCells);
void fill_particles(NEIB_PARTICLES &neibParticles, NEIB_CELLS neibCells, int *cellGrid);
void fill_particles(NEIB_PARTICLES &neibParticles, NEIB_CELLS neibCells, int *cellGrid, int subtask_id);
void set_pos_x(P_DATA_TYPE &particle, FLOAT3 r);
INT3 get_chunkIndex(int chunk);

FLOAT3 bodyBodyInteraction(P_DATA_TYPE bi, T_DATA_TYPE bj, FLOAT3 ai);
int bodyBodyCollision(P_DATA_TYPE bi, T_DATA_TYPE bj);
void init_particle(P_DATA_TYPE &particle);
void reset_particle(P_DATA_TYPE &particle);
void survive_particle(P_DATA_TYPE &particle);
void copy_particle(P_DATA_TYPE &particle, P_DATA_TYPE src_particle);

int q_remove(QUEUE_INFO* devQueueInfo, int* devQueue, int seg_type, int seg_tid);
int q_remove(QUEUE_INFO* devQueueInfo, int* devQueue, int seg_type, int seg_tid, int subtask_id);
void q_insert(QUEUE_INFO* devQueueInfo, int* devQueue, int seg_type, int seg_tid, int x);
void q_insert(QUEUE_INFO* devQueueInfo, int* devQueue, int seg_type, int seg_tid, int x, int subtask_id);

namespace particleSystem
{

using namespace pm;

#ifdef BUILD_CUDA
#include <cuda.h>
	typedef void (*particleSystem_cudaFuncPtr)(pmTaskInfo pTaskInfo, pmDeviceInfo* pDeviceInfo, pmSubtaskInfo pSubtaskInfo, pmStatus* pStatus);
	int singleGpuparticleSystem(P_DATA_TYPE* pInputX, P_DATA_TYPE* pOutputX, int pDim);

	extern particleSystem_cudaFuncPtr particleSystem_init_particles_cudaFunc;
	extern particleSystem_cudaFuncPtr particleSystem_setup_seeds_cudaFunc;
	extern particleSystem_cudaFuncPtr particleSystem_q_start_fast_cudaFunc;
	extern particleSystem_cudaFuncPtr particleSystem_init_iframe_cudaFunc;
	extern particleSystem_cudaFuncPtr particleSystem_calc_forces_cudaFunc;
	extern particleSystem_cudaFuncPtr particleSystem_pkg_distrib_cudaFunc;
	extern particleSystem_cudaFuncPtr particleSystem_build_grid_cudaFunc;
#endif

enum memIndex
{
	INDEX_MEM_PARTICLES = 0,
	INDEX_MEM_TDATA,
	INDEX_MEM_QUEUE,
	INDEX_MEM_QUEUEINFO,
	INDEX_MEM_CHUNKGRID,
	INDEX_MEM_CELLGRID,
	INDEX_MEM_PKGDISTRIB,
	INDEX_MEM_RANDSTATES,
	INDEX_MEM_GRIDMAX,
	MAX_INDEX_MEM
};

typedef struct particleSystemTaskConf
{
	int subtaskElems;
	int iteration;
#ifdef BUILD_CUDA
	pmCudaLaunchConf cudaLaunchConf;
#endif
} particleSystemTaskConf;

}

#endif

