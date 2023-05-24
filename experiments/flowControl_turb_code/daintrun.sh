NLES=32
statetype=psiomega #psiomega # [enstrophy,energy,psidiag,psiomegadiag,psiomegalocal] 
gensize=10
nagents=4
nconcurrent=3
Tspinup=1e3
Thorizon=1e4
NumRLSteps=1e3
EBRU=1

export OMP_NUM_THREADS=12
mpirun -n 4 python3 run-vracer-turb-mpi.py --statetype=${statetype} --NLES=${NLES} --gensize=${gensize} --nagents=${nagents} --nconcurrent=${nconcurrent} --Tspinup=${Tspinup} --Thorizon=${Thorizon} --numRLSteps=${NumRLSteps} --EBRU ${EBRU}
