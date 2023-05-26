NLES=32
statetype=psiomega #psiomega # [enstrophy,energy,psidiag,psiomegadiag,psiomegalocal] 
gensize=10
nagents=4
nconcurrent=$((${SLURM_NNODES}-1))
Tspinup=1e3
Thorizon=1e4
NumRLSteps=1e3
EPERU=1

echo "Number of nodes allocated:    ${SLURM_NNODES}"
echo "Number of concurrent workers: ${nconcurrent}"

export OMP_NUM_THREADS=12
srun --nodes ${SLURM_NNODES} --ntasks-per-node 1 --threads-per-core 12 python3 run-vracer-turb-mpi.py --statetype=${statetype} --NLES=${NLES} --gensize=${gensize} --nagents=${nagents} --nconcurrent=${nconcurrent} --Tspinup=${Tspinup} --Thorizon=${Thorizon} --NumRLSteps=${NumRLSteps} --EPERU ${EPERU}
