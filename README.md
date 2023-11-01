# GARel: Genetic Algorithm based Receptor-ligand interaction generator
## Boosts the drug-like features and novelty of the molecules in sampling chemical space

![overview of the architecture of GARel](/image/figure.png)
## Overview
This repository contains the source of GARel, a software for DL-based de novo drug design.


## Requirements
- Python == 3.7
- pytorch >= 1.1.0
- openbabel == 2.4.1
- RDKit == 2020.09.5
- theano == 1.0.5
- vina ==1.2.0 [README](https://autodock-vina.readthedocs.io/en/latest/docking_python.html)

if utilizing GPU accelerated model training 
- CUDA==10.2 & cudnn==7.5 




## Running GARel

### Prepare molecular dataset



### Pretrain Generator
Load sourch dataset (`https://github.com/micahwang/RELATION/tree/main/data/zinc`) and target dataset ( `./data/sars_cov2_pkis.npz`).

`python model/run_pretrain.py --epoches 150
                       --steps 5000
                       --target sars_cov2
                       --batchsize 256
                       --decive 0`


### Run GARel

`python run_gen.py --target sars_cov2`




