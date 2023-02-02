NLES=32
case=1
rewardtype=z1 # [k1,k2,k3,log,]
statetype=energy # [enstrophy,energy,psidiag,psiomegadiag,] 
actiontype=CL
gensize=10
solver=training #postproces
myoutfile=${solver}_CASE${case}_N${NLES}_R${rewardtype}_S${statetype}_A${actiontype}.out
echo ${myoutfile}>>${myoutfile}
ps>>${myoutfile}
(ls -ltr|tail -n 1)>>${myoutfile}

nohup python3 -u run-vracer-turb.py --case=${case} --rewardtype=${rewardtype} --statetype=${statetype} --actiontype=${actiontype} --NLES=${NLES} --gensize=${gensize} --solver=${solver}>> ${myoutfile}&

