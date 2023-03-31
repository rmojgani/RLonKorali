NLES=128 
case=4
rewardtype=zc # [k1,k2,k3,z1,z2,z3,ze, ... ] 
statetype=enstrophy # [enstrophy,energy,psidiag,psiomegadiag,] 
actiontype=CL
gensize=10
solver=training #postproces
nagents=4

myoutfile=${solver}_CASE${case}_N${NLES}_R${rewardtype}_S${statetype}_A${actiontype}_nAgents${nagents}.out

(echo ${myoutfile})>>${myoutfile}
(ps)>>${myoutfile}
(ls -ltr|tail -n 1)>>${myoutfile}
(top -b -n 1)>>${myoutfile}
(nvidia-smi)>>${myoutfile}

nohup python3 -u run-vracer-turb.py --case=${case} --rewardtype=${rewardtype} --statetype=${statetype} --actiontype=${actiontype} --NLES=${NLES} --gensize=${gensize} --solver=${solver} --nagents=${nagents}>> ${myoutfile}&

