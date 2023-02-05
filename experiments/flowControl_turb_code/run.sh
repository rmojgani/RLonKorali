NLES=32
case=1
rewardtype=z1 # [k1,k2,k3,log,]
statetype=enstrophy # [enstrophy,energy,psidiag,psiomegadiag,] 
actiontype=CS
gensize=10
solver=training #postproces
nagents=16
myoutfile=${solver}_CASE${case}_N${NLES}_R${rewardtype}_S${statetype}_A${actiontype}_nAgents${nagents}.out
echo ${myoutfile}>>${myoutfile}
ps>>${myoutfile}
(ls -ltr|tail -n 1)>>${myoutfile}

nohup python3 -u run-vracer-turb.py --case=${case} --rewardtype=${rewardtype} --statetype=${statetype} --actiontype=${actiontype} --NLES=${NLES} --gensize=${gensize} --solver=${solver} --nagents=${nagents}>> ${myoutfile}&

