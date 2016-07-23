all:
	cd build/linux/ && make clean && make
run:
	cd build/linux/ && mpirun -n 2 --hostfile mpi-hosts --mca mpi_preconnect_mpi 1 release/particleSystem.exe 1 6 0 99

