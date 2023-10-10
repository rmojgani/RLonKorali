date +"%d-%m-%y"
for d in _result*/ ; do
    echo "$d"
    echo 'generation:'
    ls $d/*.json | wc -l
    echo 'post count'
    ls $d/C*post/*.png | wc -l

done
#python3 -m korali.rlview --dir _result_vracerC1_N64_R_k1_State_enstrophy_Action_CL --output result.png

