/**
 * Put HOST PROTOTYPE in particleSystem.h
 */

// Do not delete: Part of commented/non-regular code [For __device__ declaration]
__device__ __host__ int get_cont_rloc(int seg_type, int seg_tid) 
{
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

	return pos;
}

__device__ __host__ int get_info_rloc(int seg_type, int seg_tid) 
{
	int pos = 0;

	switch(seg_type) {
		case 8: pos += SEG4_COUNT;
		case 4: pos += SEG2_COUNT;
		case 2: pos += SEG1_COUNT;
		case 1: break;
		default: break;
	}
	switch(seg_type) {
		case 8: pos += seg_tid; break;
		case 4: pos += seg_tid; break;
		case 2: pos += seg_tid; break;
		case 1: pos += seg_tid; break;
		default: break;
	}

	return pos;
}

__device__ __host__ INT3 get_cell_info(int cell) 
{
	INT3 cell_info;

	int i3   = cell / (GRID_DIM*GRID_DIM);
	int i3_r = cell % (GRID_DIM*GRID_DIM);
	int i1   = i3_r / GRID_DIM;
	int i2   = i3_r % GRID_DIM;

	int q1 = floor((i1*1.0)/CHUNK_DIM);
	int q2 = floor((i2*1.0)/CHUNK_DIM);
	int q3 = floor((i3*1.0)/CHUNK_DIM);

	int chunk = q3*CHUNK_FACTOR*CHUNK_FACTOR + q1*CHUNK_FACTOR + q2;

	int r1 = i1 % CHUNK_DIM;
	int r2 = i2 % CHUNK_DIM;
	int r3 = i3 % CHUNK_DIM;

	int seg_type, seg_type_1, seg_type_2, seg_type_3;
	int seg_tid, seg_tid_1, seg_tid_2, seg_tid_3;

	if(r1 == 0) {
		seg_type_1 = 2;
		seg_tid_1 = q1;
	}
	else if(r1 == CHUNK_DIM-1) {
		seg_type_1 = 2;
		seg_tid_1 = q1+1;
	}
	else {
		seg_type_1 = 1;
		seg_tid_1 = q1;
	}

	if(r2 == 0) {
		seg_type_2 = 2;
		seg_tid_2 = q2;
	}
	else if(r2 == CHUNK_DIM-1) {
		seg_type_2 = 2;
		seg_tid_2 = q2+1;
	}
	else {
		seg_type_2 = 1;
		seg_tid_2 = q2;
	}

	if(r3 == 0) {
		seg_type_3 = 2;
		seg_tid_3 = q3;
	}
	else if(r3 == CHUNK_DIM-1) {
		seg_type_3 = 2;
		seg_tid_3 = q3+1;
	}
	else {
		seg_type_3 = 1;
		seg_tid_3 = q3;
	}

	seg_type = seg_type_1 * seg_type_2 * seg_type_3;
	seg_tid = -1;

	if(seg_type == 1) {
		seg_tid = seg_tid_3*CHUNK_FACTOR*CHUNK_FACTOR + seg_tid_1*CHUNK_FACTOR + seg_tid_2;
	}
	else if(seg_type == 2) {
		if(seg_type_3 == 2) {
			seg_tid = seg_tid_3*(CHUNK_FACTOR*CHUNK_FACTOR + 2*CHUNK_FACTOR*(CHUNK_FACTOR+1) ) + seg_tid_1*CHUNK_FACTOR + seg_tid_2;
		}
		else if(seg_type_2 == 2) {
			seg_tid = (seg_tid_3+1)*(CHUNK_FACTOR*CHUNK_FACTOR) + seg_tid_3*(2*CHUNK_FACTOR*(CHUNK_FACTOR+1) ) + (seg_tid_1+1)*CHUNK_FACTOR + seg_tid_1*(CHUNK_FACTOR+1) + seg_tid_2;
		}
		else if(seg_type_1 == 2) {
			seg_tid = (seg_tid_3+1)*(CHUNK_FACTOR*CHUNK_FACTOR) + seg_tid_3*(2*CHUNK_FACTOR*(CHUNK_FACTOR+1) ) + seg_tid_1*(CHUNK_FACTOR+CHUNK_FACTOR+1) + seg_tid_2;
		}
	}
	else if(seg_type == 4) {
		if(seg_type_3 == 1) {
			seg_tid = (seg_tid_3+1)*(2*CHUNK_FACTOR*(CHUNK_FACTOR+1)) + seg_tid_3*((CHUNK_FACTOR+1)*(CHUNK_FACTOR+1)) + seg_tid_1*(CHUNK_FACTOR+1) + seg_tid_2;
		}
		else if(seg_type_2 == 1) {
			seg_tid = (seg_tid_3)*(2*CHUNK_FACTOR*(CHUNK_FACTOR+1)) + seg_tid_3*((CHUNK_FACTOR+1)*(CHUNK_FACTOR+1)) + seg_tid_1*(2*CHUNK_FACTOR+1) + seg_tid_2;
		}
		else if(seg_type_1 == 1) {
			seg_tid = (seg_tid_3)*(2*CHUNK_FACTOR*(CHUNK_FACTOR+1)) + seg_tid_3*((CHUNK_FACTOR+1)*(CHUNK_FACTOR+1)) + seg_tid_1*(2*CHUNK_FACTOR+1) + CHUNK_FACTOR + seg_tid_2;
		}
	}
	else if(seg_type == 8) {
		seg_tid = (seg_tid_3)*((CHUNK_FACTOR+1)*(CHUNK_FACTOR+1)) + seg_tid_1*(CHUNK_FACTOR+1) + seg_tid_2;
	}

	cell_info.a = chunk;
	cell_info.b = seg_type;
	cell_info.c = seg_tid;

	return cell_info;
}

__device__ __host__ void set_pkg_segments(int chunk, PAIR *seg_list)
{

	// won't work if some segments (seg1, seg2, seg4, seg8) are missing [Due to insufficient chunk dim]
	// No problem if dimensions are big (not trivially small)

	int i1, i2, i3, i3_r;
	int r, s, t;
	int t1;
	int t2_1, t2_2, t2_3, t2_4, t2_5, t2_6;
	int t4_1, t4_2, t4_3, t4_4, t4_5, t4_6, t4_7, t4_8, t4_9, t4_10, t4_11, t4_12;
	int t8_1, t8_2, t8_3, t8_4, t8_5, t8_6, t8_7, t8_8;

	i3   = chunk / (CHUNK_FACTOR*CHUNK_FACTOR);
	i3_r = chunk % (CHUNK_FACTOR*CHUNK_FACTOR);
	i1   = i3_r / CHUNK_FACTOR;
	i2   = i3_r % CHUNK_FACTOR;

	r = (CHUNK_FACTOR+1)*(CHUNK_FACTOR+1);
	s = (CHUNK_FACTOR+1)*CHUNK_FACTOR;
	t = CHUNK_FACTOR*CHUNK_FACTOR;

	t8_1 = i3*r + i1*(CHUNK_FACTOR+1) + i2;
	t8_2 = t8_1 + 1;
	t8_3 = t8_1 + CHUNK_FACTOR+1;
	t8_4 = t8_3 + 1;
	t8_5 = t8_1 + r;
	t8_6 = t8_2 + r;
	t8_7 = t8_3 + r;
	t8_8 = t8_4 + r;

	t4_1 = i3*(2*s + r) + i1*(2*CHUNK_FACTOR+1) + i2;
	t4_2 = t4_1 + CHUNK_FACTOR;
	t4_3 = t4_2 + 1;
	t4_4 = t4_3 + CHUNK_FACTOR;
	t4_5 = i3*(2*s + r) + 2*s + i1*(CHUNK_FACTOR+1) + i2;
	t4_6 = t4_5 + 1;
	t4_7 = t4_6 + CHUNK_FACTOR;
	t4_8 = t4_7 + 1;
	t4_9  = t4_1 + (2*s + r);
	t4_10 = t4_2 + (2*s + r);
	t4_11 = t4_3 + (2*s + r);
	t4_12 = t4_4 + (2*s + r);

	t2_1 = i3*(t + 2*s) + i1*CHUNK_FACTOR + i2;
	t2_2 = i3*(t + 2*s) + t + i1*(2*CHUNK_FACTOR+1) + i2;
	t2_3 = t2_2 + CHUNK_FACTOR;
	t2_4 = t2_3 + 1;
	t2_5 = t2_4 + CHUNK_FACTOR;
	t2_6 = t2_1 + (t + 2*s);

	t1 = chunk;

	seg_list[0].c = 1; seg_list[0].p = t1;
	seg_list[1].c = 2; seg_list[1].p = t2_1;
	seg_list[2].c = 2; seg_list[2].p = t2_2;
	seg_list[3].c = 2; seg_list[3].p = t2_3;
	seg_list[4].c = 2; seg_list[4].p = t2_4;
	seg_list[5].c = 2; seg_list[5].p = t2_5;
	seg_list[6].c = 2; seg_list[6].p = t2_6;
	seg_list[7].c = 4; seg_list[7].p = t4_1;
	seg_list[8].c = 4; seg_list[8].p = t4_2;
	seg_list[9].c = 4; seg_list[9].p = t4_3;
	seg_list[10].c = 4; seg_list[10].p = t4_4;
	seg_list[11].c = 4; seg_list[11].p = t4_5;
	seg_list[12].c = 4; seg_list[12].p = t4_6;
	seg_list[13].c = 4; seg_list[13].p = t4_7;
	seg_list[14].c = 4; seg_list[14].p = t4_8;
	seg_list[15].c = 4; seg_list[15].p = t4_9;
	seg_list[16].c = 4; seg_list[16].p = t4_10;
	seg_list[17].c = 4; seg_list[17].p = t4_11;
	seg_list[18].c = 4; seg_list[18].p = t4_12;
	seg_list[19].c = 8; seg_list[19].p = t8_1;
	seg_list[20].c = 8; seg_list[20].p = t8_2;
	seg_list[21].c = 8; seg_list[21].p = t8_3;
	seg_list[22].c = 8; seg_list[22].p = t8_4;
	seg_list[23].c = 8; seg_list[23].p = t8_5;
	seg_list[24].c = 8; seg_list[24].p = t8_6;
	seg_list[25].c = 8; seg_list[25].p = t8_7;
	seg_list[26].c = 8; seg_list[26].p = t8_8;

	return;
}

/**************************************************************/

__device__ __host__ FLOAT3 bodyBodyInteraction(P_DATA_TYPE bi, T_DATA_TYPE bj, FLOAT3 ai)
{
        FLOAT3 r;

	if( (bi.age < KID_AGE) 
	    || (bj.age < KID_AGE) ) {
		return ai;
	}

        // r_ij
        r.x = bj.x - bi.x;
        r.y = bj.y - bi.y;
        r.z = bj.z - bi.z;

        // distSqr = dot(r_ij, r_ij) + EPS^2
        float distSq  = r.x * r.x + r.y * r.y + r.z * r.z;
        float distSqr = distSq + EPS2;

        // invDistCube = 1/(distSqr^3)
        float distSixth = distSqr * distSqr * distSqr;
        float invDistCube = 1.0f/sqrtf(distSixth);

        // s = m_j * invDistCube
        float s = bj.w * invDistCube;

        // a_i = a_i + s * r_ij
        ai.x += r.x * s;
        ai.y += r.y * s;
        ai.z += r.z * s;

        return ai;
}

__device__ __host__ int bodyBodyCollision(P_DATA_TYPE bi, T_DATA_TYPE bj)
{
        float3 r;

        // r_ij
        r.x = bj.x - bi.x;
        r.y = bj.y - bi.y;
        r.z = bj.z - bi.z;

	// distance
	float dist = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z);

	// no collision if collision criteria not satisfied
	if( (dist > COLLISION_RADIUS) 
	    || (bi.age < KID_AGE) 
	    || (bj.age < KID_AGE) ) {
		return 0;
	}

	// no collision if any particle is old (going to die)
	if( bi.age > PARTICLE_LIFE || bj.age > PARTICLE_LIFE ) {
		return 0;
	}

	if( bi.id > bj.id) {
		return 1; // survive
	}
	else if( bi.id < bj.id) {
		return 2; // kill
	}

	return 0;
}

/**************************************************************/

__device__ __host__ int q_remove(QUEUE_INFO* devQueueInfo, int* devQueue, int seg_type, int seg_tid, int subtask_id)
{
	int iloc, pos;
	int item;

	iloc = get_info_rloc(seg_type, seg_tid);

	if(devQueueInfo[iloc].count <= 0) {
//		printf("Queue underflow.\n");
		return -1;
	}

	pos = devQueueInfo[iloc].front;

	if(devQueueInfo[iloc].count == 1) {
		devQueueInfo[iloc].front = -1;
		devQueueInfo[iloc].rear  = -1;
	}
	else {
		//next
		if (devQueueInfo[iloc].front == (devQueueInfo[iloc].rloc + devQueueInfo[iloc].seg_size - 1)) devQueueInfo[iloc].front = devQueueInfo[iloc].rloc;
		else devQueueInfo[iloc].front++;
	}

	devQueueInfo[iloc].count--;

	if(subtask_id >= 0) {
		pos = get_natural_pos(pos, subtask_id);
	}

	item = devQueue[pos];
	devQueue[pos] = -1;

	return item;
}

__device__ __host__ int q_remove(QUEUE_INFO* devQueueInfo, int* devQueue, int seg_type, int seg_tid)
{
	return q_remove(devQueueInfo, devQueue, seg_type, seg_tid, -1);
}

__device__ __host__ void q_insert(QUEUE_INFO* devQueueInfo, int* devQueue, int seg_type, int seg_tid, int x, int subtask_id)
{
	int iloc, pos;

	iloc = get_info_rloc(seg_type, seg_tid);

	if(devQueueInfo[iloc].count == devQueueInfo[iloc].seg_size) {
//		printf("Queue overflow.\n");
		return;
	}

	if(devQueueInfo[iloc].count == 0) {
		devQueueInfo[iloc].front = devQueueInfo[iloc].rloc;
		devQueueInfo[iloc].rear  = devQueueInfo[iloc].rloc;
	}
	else {
		//next
		if (devQueueInfo[iloc].rear == (devQueueInfo[iloc].rloc + devQueueInfo[iloc].seg_size - 1)) devQueueInfo[iloc].rear = devQueueInfo[iloc].rloc;
		else devQueueInfo[iloc].rear++;
	}

	devQueueInfo[iloc].count++;

	pos = devQueueInfo[iloc].rear;

	if(subtask_id >= 0) {
		pos = get_natural_pos(pos, subtask_id);
	}

	devQueue[pos] = x;
}

void q_insert(QUEUE_INFO* devQueueInfo, int* devQueue, int seg_type, int seg_tid, int x)
{
	q_insert(devQueueInfo, devQueue, seg_type, seg_tid, x, -1);
}

__device__ int q_remove(int t, QUEUE_INFO* devQueueInfo, int* devQueue, int seg_type, int seg_tid, int subtask_id)
{
	int iloc = get_info_rloc(seg_type, seg_tid);
	int d = -1;

	int laneid = (t & (WARP_SIZE-1)); // NOTE: Warp size = 32
	for( int i=0; i < WARP_SIZE; i++) {
	if( i == laneid ) {

		while( atomicCAS(&(devQueueInfo[iloc].lock), 0, 1) ); // Lock Queue

		d = q_remove(devQueueInfo, devQueue, seg_type, seg_tid, subtask_id);

		atomicExch(&(devQueueInfo[iloc].lock), 0);  // Unlock Queue
	}
	}

	return d;
}

__device__ int q_remove(int t, QUEUE_INFO* devQueueInfo, int* devQueue, int seg_type, int seg_tid)
{
	return q_remove(t, devQueueInfo, devQueue, seg_type, seg_tid, -1);
}

__device__ void q_insert(int t, QUEUE_INFO* devQueueInfo, int* devQueue, int seg_type, int seg_tid, int x, int subtask_id)
{
	int iloc = get_info_rloc(seg_type, seg_tid);

	int laneid = (t & (WARP_SIZE-1)); // NOTE: Warp size = 32
	for( int i=0; i < WARP_SIZE; i++) {
	if( i == laneid ) {

		while( atomicCAS(&(devQueueInfo[iloc].lock), 0, 1) ); // Lock Queue

		q_insert(devQueueInfo, devQueue, seg_type, seg_tid, x, subtask_id);

		atomicExch(&(devQueueInfo[iloc].lock), 0);  // Unlock Queue
	
	}
	}
}

__device__ void q_insert(int t, QUEUE_INFO* devQueueInfo, int* devQueue, int seg_type, int seg_tid, int x)
{
	q_insert(t, devQueueInfo, devQueue, seg_type, seg_tid, x, -1);
}


