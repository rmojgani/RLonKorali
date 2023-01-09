#rm *.out
#rm *.png; rm N*_t\=*.mat; 
#rm -rf _result_vracer/*
#rm N*_t\=*.out
nohup python3 run-vracer-turbpost.py  --case=4&> mykoralipost.out &
