all:
	pwd
	@echo "OOPS! cp/cc/run"
	@echo ""
cc:
	clear
	clear
	cd ~/unicorn/pmlib/testSuite/particleSystem/build/linux/ && make clean && make
	@echo ""
cc_g:
	clear
	clear
	cd ~/unicorn/pmlib/testSuite/particleSystem/build/linux/ && make DEBUG=1 clean && make DEBUG=1
	@echo ""
run:
	clear
	clear
	cd ~/unicorn/pmlib/testSuite/particleSystem/build/linux/ && mpirun -n 2 --hostfile mpi-hosts --mca mpi_preconnect_mpi 1 release/particleSystem.exe 1 6 0 99
	@echo ""
run_g:
	clear
	clear
	cd ~/unicorn/pmlib/testSuite/particleSystem/build/linux/ && mpirun -n 2 --hostfile mpi-hosts --mca mpi_preconnect_mpi 1 debug/particleSystem_g.exe 1 6 0 99
	@echo ""
runx:
	clear
	clear
	cd ~/unicorn/pmlib/testSuite/particleSystem/build/linux/ && mpirun -n 1 --hostfile mpi-hosts --mca mpi_preconnect_mpi 1 release/particleSystem.exe 1 6 0 99 2> /dev/null
	@echo ""
run_h:
	clear
	clear
	cd ~/unicorn/pmlib/testSuite/particleSystem/build/linux/ && mpirun -n 3 --hostfile mpi-hosts --mca mpi_preconnect_mpi 1 release/particleSystem.exe 1 6 0 99
	@echo ""
hpc:
	clear
	clear
	qsub pbs.sh
	@echo ""
cp:
	cd /media/abraj/C472E7A372E79906/LIVE/abhi/IITD/project/UNICORN/src/unicorn && scp -r particleSystem mcs142113@hpc.iitd.ac.in:~/unicorn/pmlib/testSuite/
	cd /media/abraj/C472E7A372E79906/LIVE/abhi/IITD/project/UNICORN/src/unicorn/particleSystem && scp makefile mcs142113@hpc.iitd.ac.in:~/
	@echo ""
clean:
	clear
	clear
	rm -r ~/unicorn/pmlib/testSuite/particleSystem
	@echo ""

#
# abhishekr@gpuserver.cse.iitd.ac.in
# mcs142113@hpca.iitd.ac.in
# mcs142113@hpc.iitd.ac.in
# mcs142113@gpu.hpc.iitd.ac.in
#
