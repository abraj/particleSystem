
__device__ __host__ int get_pkg_rloc(int seg_type, int local_tid) 
{
	int pos = 0;

	switch(seg_type) {
		case 8: pos += 12 * SEG4_SIZE_T;
		case 4: pos +=  6 * SEG2_SIZE_T;
		case 2: pos +=  1 * SEG1_SIZE_T;
		case 1: break;
		default: break;
	}
	switch(seg_type) {
		case 8: pos += local_tid * SEG8_SIZE_T; break;
		case 4: pos += local_tid * SEG4_SIZE_T; break;
		case 2: pos += local_tid * SEG2_SIZE_T; break;
		case 1: pos += local_tid * SEG1_SIZE_T; break;
		default: break;
	}

	return pos;
}

__device__ __host__ PAIR get_id_info(int id) 
{
	PAIR id_info = {-1, -1};
	int seg_type, seg_tid;
	int rid, seg_size;

	if(id >= 0 && id < SEG1_SIZE) {
		seg_type = 1;
		rid = id;
	}
	else if(id >= SEG1_SIZE && id < SEG1_SIZE + SEG2_SIZE) {
		seg_type = 2;
		rid = id - (SEG1_SIZE);
	}
	else if(id >= SEG1_SIZE + SEG2_SIZE && id < SEG1_SIZE + SEG2_SIZE + SEG4_SIZE) {
		seg_type = 4;
		rid = id - (SEG1_SIZE + SEG2_SIZE);
	}
	else if(id >= SEG1_SIZE + SEG2_SIZE + SEG4_SIZE && id < CONTAINER_SIZE) {
		seg_type = 8;
		rid = id - (SEG1_SIZE + SEG2_SIZE + SEG4_SIZE);
	}
	else {
		printf("ERROR! Invalid ID\n");
		return id_info;
	}

	switch(seg_type) {
		case 8: seg_size = SEG8_SIZE_T; break;
		case 4: seg_size = SEG4_SIZE_T; break;
		case 2: seg_size = SEG2_SIZE_T; break;
		case 1: seg_size = SEG1_SIZE_T; break;
		default: break;
	}

	seg_tid = rid / seg_size;

	id_info.c = seg_type;
	id_info.p = seg_tid;

	return id_info;
}

__device__ __host__ int get_local_pos(int particle_id, PAIR *seg_list)
{
	int pos;

	PAIR particle_info = get_id_info(particle_id);
	int seg_type = particle_info.c;
	int seg_tid  = particle_info.p;

	int t_start, t_end;
	if(seg_type == 1) { t_start = 0; t_end = 0; }
	else if(seg_type == 2) { t_start = 1; t_end = 6; }
	else if(seg_type == 4) { t_start = 7; t_end = 18; }
	else { t_start = 19; t_end = 26; }  // seg_type == 8

	int found = 0, local_tid = 0;
	for(int i=t_start; i<=t_end; i++) {
		if(seg_list[i].p == seg_tid) {
			found = 1;
			break;
		}
		local_tid++;
	}

	if(found == 0) {
		pos = -1;
	}
	else {
		int global_offset = get_cont_rloc(seg_type, seg_tid);
		int local_offset = get_pkg_rloc(seg_type, local_tid);
		pos = particle_id - global_offset + local_offset;
	}

	return pos;
}

int get_natural_pos(int particle_id, int subtask_id)
{
	int pos;

	if(particle_id >= 0) {
		int global_offset = get_cont_rloc(1, subtask_id);
		pos = particle_id - global_offset;
	}
	else {
		pos = -1;
	}

	return pos;
}

__device__ __host__ int set_pos_t(P_DATA_TYPE &particle, FLOAT3 r)
{
	int t1, t2, t3;
	FLOAT3 rt;

	rt.x = r.x;
	rt.y = r.y;
	rt.z = r.z;

	int i1 = floor((-1.0 * rt.y) / CELL_SIZE) + (GRID_DIM/2);
	int i2 = floor(( 1.0 * rt.x) / CELL_SIZE) + (GRID_DIM/2);
	int i3 = floor((-1.0 * rt.z) / CELL_SIZE) + (GRID_DIM/2);

	// relocate particle 
  	// otherwise particle may be lost
	if( !((i1 >= 0 && i1 < GRID_DIM) && (i2 >= 0 && i2 < GRID_DIM) && (i3 >= 0 && i3 < GRID_DIM)) ) {

		while( !((i1 >= 0 && i1 < GRID_DIM) && (i2 >= 0 && i2 < GRID_DIM) && (i3 >= 0 && i3 < GRID_DIM)) ) {
			if(!(i1 >= 0 && i1 < GRID_DIM)) {
				t1 = i1; i1 = (i1+GRID_DIM) % GRID_DIM;
				rt.y += -1.0*(i1-t1)*CELL_SIZE;
			}
			if(!(i2 >= 0 && i2 < GRID_DIM)) {
				t2 = i2; i2 = (i2+GRID_DIM) % GRID_DIM;
				rt.x += (i2-t2)*CELL_SIZE;
			}
			if(!(i3 >= 0 && i3 < GRID_DIM)) {
				t3 = i3; i3 = (i3+GRID_DIM) % GRID_DIM;
				rt.z += -1.0*(i3-t3)*CELL_SIZE;
			}
		}
	}

	particle.x = rt.x;
	particle.y = rt.y;
	particle.z = rt.z;

	int cell = i3*GRID_DIM*GRID_DIM + i1*GRID_DIM + i2;
	particle.cell = cell;

	return cell;
}

__device__ __host__ void set_pos_i(P_DATA_TYPE &particle, FLOAT3 r)
{
	int cell = set_pos_t(particle, r);

	INT3 cell_info = get_cell_info(cell);
	particle.chunk    = cell_info.a;

	particle.seg_type = cell_info.b;
	particle.seg_tid  = cell_info.c;
}

__device__ __host__ void set_pos_x(P_DATA_TYPE &particle, FLOAT3 r)
{
	int cell = set_pos_t(particle, r);

	INT3 cell_info = get_cell_info(cell);
	particle.chunk    = cell_info.a;

	// change in location may also involve CHANGE in segment ==> CHANGE in ID
	if( !(particle.seg_type == cell_info.b && particle.seg_tid == cell_info.c) ) {
		if( !(particle.seg_type == -1 && particle.seg_tid == -1) ) {
			particle.seg_fault = true;
		}
		particle.seg_type = cell_info.b;
		particle.seg_tid  = cell_info.c;
	}

}

__device__ __host__ void create_particle_s(P_DATA_TYPE *devX, int t, float w, float age, float fert_age, 
						float x, float y, float z, float vx, float vy, float vz)
{
	FLOAT3 r = { x, y, z };

	set_pos_i( devX[t], r);

	devX[t].w = w;
	devX[t].age = age;
	devX[t].fertility_age = fert_age;
	devX[t].is_parent = false;

	devX[t].vx = vx;
	devX[t].vy = vy;
	devX[t].vz = vz;

	devX[t].ax = 0.0f;
	devX[t].ay = 0.0f;
	devX[t].az = 0.0f;
}

//__device__ void create_particle_d(P_DATA_TYPE *devX, QUEUE *dq, int gtid, float w, float fert_age, 
//						float x, float y, float z, float vx, float vy, float vz)
__device__ void create_particle_d(P_DATA_TYPE *devX, int gtid, float w, float age, float fert_age, 
						float x, float y, float z, float vx, float vy, float vz)
{
//	int laneid = (gtid & (WARP_SIZE-1)); // NOTE: Warp size = 32
//	for( int i=0; i < WARP_SIZE; i++) {
//	if( i == laneid ) {

//		while( atomicCAS(&dq->lock, 0, 1) ); // Lock Queue

//		int t = q_remove(dq);
		int t = gtid;
		if(t != -1) {
			create_particle_s(devX, t, w, age, fert_age, x, y, z, vx, vy, vz);
		}

//		atomicExch(&dq->lock, 0);  // Unlock Queue
//	}
//	}
}

__device__ __host__ void copy_particle(P_DATA_TYPE &particle, P_DATA_TYPE src_particle)
{
	int id = particle.id;
	particle = src_particle;
	particle.id = id;
}

__device__ __host__ void reset_particle(P_DATA_TYPE &particle)
{
	particle.cell = -1;
	particle.chunk = -1;
	particle.seg_type = -1;
	particle.seg_tid = -1;

	particle.seg_fault = false;
	particle.is_parent = false;

	particle.w = 0.0f;
	particle.age = 0.0f;
	particle.fertility_age = 0.0f;

	particle.x = 0.0f;
	particle.y = 0.0f;
	particle.z = 0.0f;

	particle.vx = 0.0f;
	particle.vy = 0.0f;
	particle.vz = 0.0f;

	particle.ax = 0.0f;
	particle.ay = 0.0f;
	particle.az = 0.0f;
}

__device__ __host__ void init_particle(P_DATA_TYPE &particle)
{
	reset_particle(particle);
}

__device__ __host__ void survive_particle(P_DATA_TYPE &particle)
{
	particle.age = 0.0f;
	particle.is_parent = false;

	particle.vx = 0.0f;
	particle.vy = 0.0f;
	particle.vz = 0.0f;

	particle.ax = 0.0f;
	particle.ay = 0.0f;
	particle.az = 0.0f;
}

/*
__device__ bool do_toss(int gtid, curandState *devStates)
{
	curandState &state = devStates[gtid];
	int randInt = (int)(curand_uniform(&state) * 100) - 100/2;

	return (randInt > 0);
}
*/

__device__ float get_random_number(int gtid, curandState *devStates, float min_val, float max_val)
{
	curandState &state = devStates[gtid];
	return ( min_val + curand_uniform(&state) * (max_val-min_val) );
}

__device__ FLOAT3 get_random_uvector(int gtid, curandState *devStates)
{
	curandState &state = devStates[gtid];
	int randInt1 = (int)(curand_uniform(&state) * 100) - 100/2;
	int randInt2 = (int)(curand_uniform(&state) * 100) - 100/2;
	int randInt3 = (int)(curand_uniform(&state) * 100) - 100/2;

	FLOAT3 vec = {randInt1*1.0, randInt2*1.0, randInt3*1.0};

	float mag = sqrtf(vec.x * vec.x * 1.0 + vec.y * vec.y * 1.0 + vec.z * vec.z * 1.0);
	vec.x /= mag;
	vec.y /= mag;
	vec.z /= mag;

	return vec;
}

__device__ __host__ INT3 get_chunkIndex(int chunk)
{
	int i1,i2,i3,n;

	n = chunk;
	i3 = n/(CHUNK_FACTOR*CHUNK_FACTOR);
	n -= i3*CHUNK_FACTOR*CHUNK_FACTOR;
	i1 = n/CHUNK_FACTOR;
	n -= i1*CHUNK_FACTOR;
	i2 = n;

	INT3 p; 
	p.a = i1; p.b = i2; p.c = i3;

	return p;
}

__device__ __host__ INT3 get_cellIndex(int cell)
{
	int i1,i2,i3,n;

	n = cell;
	i3 = n/(GRID_DIM*GRID_DIM);
	n -= i3*GRID_DIM*GRID_DIM;
	i1 = n/GRID_DIM;
	n -= i1*GRID_DIM;
	i2 = n;

	INT3 p; 
	p.a = i1; p.b = i2; p.c = i3;

	return p;
}

__device__ __host__ void fill_cell(NEIB_CELLS &neibCells, int cell, INT3 p0)
{
	if(cell >= 0 && cell < GRID_DIM*GRID_DIM*GRID_DIM) {
		INT3 p = get_cellIndex(cell);

		INT3 r;
		r.a = p0.a - p.a;
		r.b = p0.b - p.b;
		r.c = p0.c - p.c;

		int distSqr = r.a * r.a + r.b * r.b + r.c * r.c;

		if(distSqr <= 3) {
			neibCells.data[neibCells.size++] = cell;
		}
	}
}

__device__ __host__ void fill_cells(NEIB_CELLS &neibCells)
{
        int mcell = neibCells.data[0];
	INT3 p0 = get_cellIndex(mcell);

        fill_cell(neibCells, mcell-1, p0);
        fill_cell(neibCells, mcell+1, p0);

        fill_cell(neibCells, mcell-1 -GRID_DIM, p0);
        fill_cell(neibCells, mcell+0 -GRID_DIM, p0);
        fill_cell(neibCells, mcell+1 -GRID_DIM, p0);

        fill_cell(neibCells, mcell-1 +GRID_DIM, p0);
        fill_cell(neibCells, mcell+0 +GRID_DIM, p0);
        fill_cell(neibCells, mcell+1 +GRID_DIM, p0);

        fill_cell(neibCells, mcell-1 -GRID_DIM -GRID_DIM*GRID_DIM, p0);
        fill_cell(neibCells, mcell+0 -GRID_DIM -GRID_DIM*GRID_DIM, p0);
        fill_cell(neibCells, mcell+1 -GRID_DIM -GRID_DIM*GRID_DIM, p0);

        fill_cell(neibCells, mcell-1 +0        -GRID_DIM*GRID_DIM, p0);
        fill_cell(neibCells, mcell+0 +0        -GRID_DIM*GRID_DIM, p0);
        fill_cell(neibCells, mcell+1 +0        -GRID_DIM*GRID_DIM, p0);

        fill_cell(neibCells, mcell-1 +GRID_DIM -GRID_DIM*GRID_DIM, p0);
        fill_cell(neibCells, mcell+0 +GRID_DIM -GRID_DIM*GRID_DIM, p0);
        fill_cell(neibCells, mcell+1 +GRID_DIM -GRID_DIM*GRID_DIM, p0);

        fill_cell(neibCells, mcell-1 -GRID_DIM +GRID_DIM*GRID_DIM, p0);
        fill_cell(neibCells, mcell+0 -GRID_DIM +GRID_DIM*GRID_DIM, p0);
        fill_cell(neibCells, mcell+1 -GRID_DIM +GRID_DIM*GRID_DIM, p0);

        fill_cell(neibCells, mcell-1 +0        +GRID_DIM*GRID_DIM, p0);
        fill_cell(neibCells, mcell+0 +0        +GRID_DIM*GRID_DIM, p0);
        fill_cell(neibCells, mcell+1 +0        +GRID_DIM*GRID_DIM, p0);

        fill_cell(neibCells, mcell-1 +GRID_DIM +GRID_DIM*GRID_DIM, p0);
        fill_cell(neibCells, mcell+0 +GRID_DIM +GRID_DIM*GRID_DIM, p0);
        fill_cell(neibCells, mcell+1 +GRID_DIM +GRID_DIM*GRID_DIM, p0);
}

__device__ __host__ void fill_cell_particles(NEIB_PARTICLES &neibParticles, int cell, int *cellGrid, int subtask_id)
{
	int c = (1+MAX_PARTICLES_PER_CELL);
	int i = cell;

	if(subtask_id >= 0) {
		INT3 p = get_chunkIndex(subtask_id);
		int global_offset = GRID_DIM*GRID_DIM*(p.c*CHUNK_DIM) + GRID_DIM*(p.a*CHUNK_DIM) + (p.b*CHUNK_DIM);
		i = i - global_offset;
	}

	int count = cellGrid[i*c+0];
	if(count > 0) {
		for(int t = 1; t <= count; t++) {
			if(neibParticles.size < MAX_NEIB_PARTICLES) {
				neibParticles.data[neibParticles.size++] = cellGrid[i*c+t];
			}
			else {
				printf(">>>>>>>>>>>>>>>>>>>>>>>> ERROR: MAX_NEIB_PARTICLES overflow! <<<<<<<<<<<<<<<<<<<<<<<<<< \n");
				//asm("trap;"); // asm("exit;");
				exit(1);
			}
		}
	}
}

__device__ __host__ void fill_cell_particles(NEIB_PARTICLES &neibParticles, int cell, int *cellGrid)
{
	fill_cell_particles(neibParticles, cell, cellGrid, -1);
}

__device__ __host__ void fill_particles(NEIB_PARTICLES &neibParticles, NEIB_CELLS neibCells, int *cellGrid, int subtask_id)
{
	for(int i=0; i<neibCells.size; i++) {
		fill_cell_particles( neibParticles, neibCells.data[i], cellGrid, subtask_id);
	}
}

__device__ __host__ void fill_particles(NEIB_PARTICLES &neibParticles, NEIB_CELLS neibCells, int *cellGrid)
{
	fill_particles(neibParticles, neibCells, cellGrid, -1);
}

