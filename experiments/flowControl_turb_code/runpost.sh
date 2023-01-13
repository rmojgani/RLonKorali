NLES=128
case=4
rewardtype=k1
statetype=psiomegadiag
actiontype=CL
solver=postprocess
myoutfile=${solver}_CASE${case}_${NLES}_R${rewardtype}_S${statetype}_A${actiontype}.out
echo ${myoutfile}>${myoutfile}
ps>>${myoutfile}
(ls -ltr|tail -n 1)>>${myoutfile}
nohup python3 run-vracer-turbpost.py --case=$case --rewardtype=$rewardtype --statetype=${statetype} --actiontype=${actiontype} --NLES=${NLES} --solver=${solver}&>>${myoutfile}&



