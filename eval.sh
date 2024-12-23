echo "Eval Path: "$1

python evaluate_model.py --algorithm GDG --test-env 0 --seed 0 --output-log $1 --checkpoint results/$1/testenv0/checkpoints/best_$2.pt &&
python evaluate_model.py --algorithm GDG --test-env 1 --seed 0 --output-log $1 --checkpoint results/$1/testenv1/checkpoints/best_$3.pt &&
python evaluate_model.py --algorithm GDG --test-env 2 --seed 0 --output-log $1 --checkpoint results/$1/testenv2/checkpoints/best_$4.pt &&
python evaluate_model.py --algorithm GDG --test-env 3 --seed 0 --output-log $1 --checkpoint results/$1/testenv3/checkpoints/best_$5.pt