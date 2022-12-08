rm *.out
rm *.png; rm N*_t\=*.mat; rm -rf _result_vracer/*
python3 run-vracer-turb.py --case='A1'
