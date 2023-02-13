#NLES=32
for NLES in 16 32 64; do

case=1
rewardtype=ke #z1 # [k1,k2,k3,ke,z1,z2,z3,ze]
statetype=energy # [enstrophy,energy,psidiag,psiomegadiag,] 
actiontype=CL
gensize=10
solver=training #postprocess
myoutfile=${solver}_CASE${case}_N${NLES}_R${rewardtype}_S${statetype}_A${actiontype}.out
echo ${myoutfile}>>${myoutfile}
ps>>${myoutfile}
(ls -ltr|tail -n 1)>>${myoutfile}

nohup python3 -u run-vracer-turb.py --case=${case} --rewardtype=${rewardtype} --statetype=${statetype} --actiontype=${actiontype} --NLES=${NLES} --gensize=${gensize} --solver=${solver}>>${myoutfile}&

BACK_PID=$!
wait $BACK_PID

done
#sudo shutdown +5
