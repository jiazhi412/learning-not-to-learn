python main.py -e debug --color_var 0.02

python main_IMDB.py -e IMDB_1005 --IMDB_train_mode eb1 --IMDB_test_mode eb2
python main_IMDB.py -e IMDB_1005 --IMDB_train_mode eb1_ex --IMDB_test_mode eb2_ex

python main_IMDB.py -e debug --IMDB_train_mode eb2 --IMDB_test_mode eb1
python main_IMDB.py -e debug --IMDB_train_mode eb1_ex --IMDB_test_mode eb2_ex
python main_IMDB.py -e debug --IMDB_train_mode eb2_ex --IMDB_test_mode eb1_ex


# Type I Bias
python main_Diabetes.py --e 321_I_2 --bias_type I --minority young --minority_size 100 