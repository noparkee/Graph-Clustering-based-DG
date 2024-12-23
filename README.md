# GDG
[ICPR 2024] Clustering-based Image-Text Graph Matching for Domain Generalization

# Graph-based-DG

## Set Environment
```shell
conda create -n GDG python=3.8

conda activate GDG

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# install dgl package
pip install dgl-1.0.1+cu113-cp38-cp38-manylinux1_x86_64.whl
pip install dglgo==0.0.2

pip install pycocotools tensorboard matplotlib wilds
```

## CUB-DG
### Dataset
You can download the **CUB-DG** dataset at [here](https://github.com/mswzeus/GVRT.git)!


### How to Train?
```shell
./train.sh [LOG_DIR_PATH] [arg_OPTIONS]
```
- If you want to train other algorithms, you should edit the `train.sh` file.
   

### How to Evaluate?
```shell
./eval.sh [LOG_DIR_PATH] [env0_CHECKPOINT] [env1_CHECKPOINT] [env2_CHECKPOINT] [env3_CHECKPOINT]
```
- If you want to evaluate other algorithms, you should edit the `eval.sh` file.

## Domainbed
We modified the DomainBed repo to utilize natural language.   
You have to download Domainbed dataset first.
For more information, please see the original repo.
