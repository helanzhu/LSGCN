

1.Recommended Environment:

numpy 1.18.1
tensorflow 1.14.0


2.Command for different datasets:

(1)PeMS04:
python main.py --n_route 307 --file_path ./datasets/PeMS04/ --graph W_307.csv --feature V_307.csv --train_days 47 --validation_days 6 --test_days 6

(2)PeMS07(Default Dataset):
python main.py --n_route 228 --file_path ./datasets/PeMS07/ --graph W_228.csv --feature V_228.csv --train_days 34 --validation_days 5 --test_days 5

(3)PeMS08:
python main.py --n_route 170 --file_path ./datasets/PeMS08/ --graph W_170.csv --feature V_170.csv --train_days 50 --validation_days 6 --test_days 6
