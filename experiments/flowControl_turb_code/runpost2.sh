#rm *.out
#rm *.png; rm N*_t\=*.mat; 
#rm -rf _result_vracer/*
#rm N*_t\=*.out
nohup python3 run-vracer-turbpost2.py  --case=4&> mykoralipost_2.out &
