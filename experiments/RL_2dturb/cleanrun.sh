rm *.out
rm *.png; rm N*_t\=*.mat; rm -rf _result_vracer/*
rm N*_t\=*.out
python3 run-vracer-turb.py --case=1
