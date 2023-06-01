NLES=32 #64
case=1
rewardtype=z1 # [k1,k2,k3,log,]
statetype=invariantlocalandglobalz #psiomega # [enstrophy,energy,psidiag,psiomegadiag,psiomegalocal,omegalocal] 
actiontype=CL
gensize=10
solver=training #postproces
nagents=16
nconcurrent=1
IF_REWARD_CUM=1 #{0,1}
Tspinup=1e4
Thorizon=1e4
NumRLSteps=1e4
EPERU=1

myoutfile=${solver}_CASE${case}_N${NLES}_R${rewardtype}_S${statetype}_A${actiontype}_nAgents${nagents}_nCCjobs${nconcurrent}_CReward${IF_REWARD_CUM}_Ts${Tspinup}_Thor${Thorizon}_NumRLSteps${NumRLSteps}_EPERU${EPERU}.out

(echo ${myoutfile})>>${myoutfile}
(ps)>>${myoutfile}
(ls -ltr|tail -n 1)>>${myoutfile}
(top -b -n 1)>>${myoutfile}
(cat /proc/meminfo | grep Mem | head -n 3)>>${myoutfile}
(nvidia-smi)>>${myoutfile}


export OMP_NUM_THREADS=8

nohup python3 -u run-vracer-turb.py --case=${case} --rewardtype=${rewardtype} --statetype=${statetype} --actiontype=${actiontype} --NLES=${NLES} --gensize=${gensize} --solver=${solver} --nagents=${nagents} --nconcurrent=${nconcurrent} --IF_REWARD_CUM=${IF_REWARD_CUM} --Tspinup=${Tspinup} --Thorizon=${Thorizon} --NumRLSteps=${NumRLSteps} --EPERU=${EPERU}>>${myoutfile}&
