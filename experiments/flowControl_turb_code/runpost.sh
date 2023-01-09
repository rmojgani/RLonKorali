#rm *.out
#rm *.png; rm N*_t\=*.mat; 
#rm -rf _result_vracer/*
#rm N*_t\=*.out
NLES=64
case=1
rewardtype=k1
statetype=psiomegadiag
actiontype=CL
solver=postprocess

nohup python3 run-vracer-turbpost.py  --case=$case --rewardtype=$rewardtype --statetype=${statetype} --actiontype=${actiontype} --NLES=${NLES} --solver=${solver}&> ${solver}_CASE${case}_R${rewardtype}_S${psiomegadiag}_A${actiontype}.out&
