echo "Output Path: "$1

python3 train_model.py --algorithm GDG --test-env 3 --seed 0 --output-log $1 $2 &&
python3 train_model.py --algorithm GDG --test-env 2 --seed 0 --output-log $1 $2 &&
python3 train_model.py --algorithm GDG --test-env 1 --seed 0 --output-log $1 $2 &&
python3 train_model.py --algorithm GDG --test-env 0 --seed 0 --output-log $1 $2