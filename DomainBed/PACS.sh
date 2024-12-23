python3 -m domainbed.scripts.train \
        --data_dir=data \
        --algorithm $1 \
        --dataset PACS \
        --trial_seed $3 \
        --hparams_seed $4 \
        --test_env 0 \
        --output_dir=train_output/PACS/$2/test_env0 $5 &&

python3 -m domainbed.scripts.train \
        --data_dir=data \
        --algorithm $1 \
        --dataset PACS \
        --trial_seed $3 \
        --hparams_seed $4 \
        --test_env 1 \
        --output_dir=train_output/PACS/$2/test_env1 $5 &&

python3 -m domainbed.scripts.train \
        --data_dir=data \
        --algorithm $1 \
        --dataset PACS \
        --trial_seed $3 \
        --hparams_seed $4 \
        --test_env 2 \
        --output_dir=train_output/PACS/$2/test_env2 $5 &&

python3 -m domainbed.scripts.train \
        --data_dir=data \
        --algorithm $1 \
        --dataset PACS \
        --trial_seed $3 \
        --hparams_seed $4 \
        --test_env 3 \
        --output_dir=train_output/PACS/$2/test_env3 $5