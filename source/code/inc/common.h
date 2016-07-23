
#ifndef __COMMON_H__
#define __COMMON_H__

/*----------------------------------------------------*/

#define SERIAL_MODE 0	// set ST_MAX 1, mpirun -n 1
#define HOST_MODE 1

#define ST_MAX 8	// 1 for serial mode; otherwise 8 or 16

#define MAX_PARTICLES_NUM 1024*1024
#define X_FACTOR 2		// expansion factor for reserve space				// GOOD: 2

#define MAX_NEIB_CELLS 27	// 3 x 3 x 3
#define BLOCK_SIZE 512		// must be a power of 2;	Minimum of 512/1024 (for Compute 2.x/3.x)
#define WARP_SIZE 32		// must be a power of 2
#define T_FACTOR 4

#define MAX_NEIB_PARTICLES (MAX_PARTICLES_PER_CELL*MAX_NEIB_CELLS)
#define MAX_PARTICLES_PER_CHUNK (MAX_PARTICLES_PER_CELL*NUM_CELLS_PER_CHUNK)
#define MAX_PARTICLES_PER_CELL (((MAX_PARTICLES_NUM)/NUM_CELLS + 1)*X_FACTOR)

#define NUM_FRAGS (NUM_CHUNKS)	// custom number of subtasks
#define NUM_CELLS (GRID_DIM*GRID_DIM*GRID_DIM)	// total number of cells
#define NUM_CHUNKS (CHUNK_FACTOR*CHUNK_FACTOR*CHUNK_FACTOR)	// total number of chunks
#define NUM_CELLS_PER_CHUNK (CHUNK_DIM*CHUNK_DIM*CHUNK_DIM)	// total number of cells per chunk
#define GRID_DIM (CHUNK_FACTOR*CHUNK_DIM)	// grid dimension in terms of cell units
#define CHUNK_FACTOR 4	//EVEN!	// number of chunks along an axis (must be even, so that GRID_DIM is even)	// GOOD: 4 [2]
#define CHUNK_DIM 4		// chunk dimension in terms of cell units					// GOOD: 6 [4]

#define CONTAINER_SIZE (SEG1_SIZE + SEG2_SIZE + SEG4_SIZE + SEG8_SIZE)
#define QUEUE_INFO_SIZE (SEG1_COUNT + SEG2_COUNT + SEG4_COUNT + SEG8_COUNT)

#define SEG1_SIZE   (SEG1_COUNT*SEG1_SIZE_T)
#define SEG1_SIZE_T (SEG1_CELLS*MAX_PARTICLES_PER_CELL)
#define SEG1_COUNT  (CHUNK_FACTOR*CHUNK_FACTOR*CHUNK_FACTOR)
#define SEG1_CELLS  ((CHUNK_DIM-2)*(CHUNK_DIM-2)*(CHUNK_DIM-2))
#define SEG2_SIZE   (SEG2_COUNT*SEG2_SIZE_T)
#define SEG2_SIZE_T (SEG2_CELLS*MAX_PARTICLES_PER_CELL)
#define SEG2_COUNT  (3*CHUNK_FACTOR*CHUNK_FACTOR*(CHUNK_FACTOR+1))
#define SEG2_CELLS  (2*(CHUNK_DIM-2)*(CHUNK_DIM-2))
#define SEG4_SIZE   (SEG4_COUNT*SEG4_SIZE_T)
#define SEG4_SIZE_T (SEG4_CELLS*MAX_PARTICLES_PER_CELL)
#define SEG4_COUNT  (3*CHUNK_FACTOR*(CHUNK_FACTOR+1)*(CHUNK_FACTOR+1))
#define SEG4_CELLS  (4*(CHUNK_DIM-2))
#define SEG8_SIZE   (SEG8_COUNT*SEG8_SIZE_T)
#define SEG8_SIZE_T (SEG8_CELLS*MAX_PARTICLES_PER_CELL)
#define SEG8_COUNT  ((CHUNK_FACTOR+1)*(CHUNK_FACTOR+1)*(CHUNK_FACTOR+1))
#define SEG8_CELLS  8

#define CELL_SIZE 5.0
#define EPS2 0.2
#define COLLISION_RADIUS 0.4
#define PARTICLE_WEIGHT_DEFAULT 60.0
#define RAND_SEED 1

#define PARTICLE_LIFE (300*DT)
#define KID_AGE ((PARTICLE_LIFE)/10.0)
#define MIN_FERTILITY_AGE ((PARTICLE_LIFE)/6.0)
#define MAX_FERTILITY_AGE ((PARTICLE_LIFE)*2.0)
#define MIN_ADULT_AGE ((PARTICLE_LIFE)/7.0)
#define MAX_ADULT_AGE ((PARTICLE_LIFE)/2.0)

#define MAX_DX (CELL_SIZE)
#define MAX_V 10.0
#define EXPLOSION_SPEED 3.0

#define DT 0.05
#define NUM_ITERATIONS 10

/*----------------------------------------------------*/

// ST_MAX*iteration for KERNEL_HEADER also

#define KERNEL_HEADER \
	int subtask_id = (int)(pSubtaskInfo.subtaskId); \
	particleSystemTaskConf* lTaskConf = (particleSystemTaskConf*)(pTaskInfo.taskConf); \
	int subtask_elems = (int)(lTaskConf->subtaskElems); \
	int tid = blockDim.x * blockIdx.x + threadIdx.x; \
	if(tid >= subtask_elems) return; \
	subtask_id = subtask_id; 		// to suppress warning [unused variable]

#define HOST_HEADER \
	int subtask_id = (int)(pSubtaskInfo.subtaskId); \
	particleSystemTaskConf* lTaskConf = (particleSystemTaskConf*)(pTaskInfo.taskConf); \
	int subtask_elems = (int)(lTaskConf->subtaskElems); \
	int iteration = (int)(lTaskConf->iteration); \
	subtask_id = ST_MAX*iteration+subtask_id; \
	subtask_elems = subtask_elems;

/*----------------------------------------------------*/

struct P_DATA_TYPE
{
	int id;
	int cell;
	int chunk;
	int seg_type;
	int seg_tid;

	bool seg_fault;
	bool is_parent;

	float w;
	float age;
	float fertility_age;

	float x;
	float y;
	float z;

	float vx;
	float vy;
	float vz;

	float ax;
	float ay;
	float az;
};

struct T_DATA_TYPE
{
	int id;

	float x;
	float y;
	float z;

	float w;
	float age;
};

struct QUEUE_INFO
{
	int front; int rear;  // 8 bytes
	int count; int lock;  // 8 bytes
	int rloc; int seg_size;  // 8 bytes
};

struct PAIR
{
	int c;
	int p;
};

struct INT3
{
	int a;
	int b;
	int c;
};

struct FLOAT3
{
	float x;
	float y;
	float z;
};

struct INT4
{
	int a;
	int b;
	int c;
	int d;
};

struct FLOAT4
{
	float x;
	float y;
	float z;
	float w;
};

struct NEIB_CELLS
{
	int data[MAX_NEIB_CELLS];
	int size;
};

struct NEIB_PARTICLES
{
	int data[MAX_NEIB_PARTICLES];
	int size;
};

struct ACC3
{
	int pid;
	float ax;
	float ay;
	float az;
};

#endif

