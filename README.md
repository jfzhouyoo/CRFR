# CRFR
> The implementation of our paper accepted by EMNLP 2021: [**CRFR: Improving Conversational Recommender Systems via Flexible Fragments Reasoning on Knowledge Graphs**](https://aclanthology.org/2021.emnlp-main.355/)

> The code and data are developed based on KGSF. Due to time reasons, it has not been transformed into a standardized code framework, so it seems that the code is bad.

<img src="https://img.shields.io/badge/Venue-EMNLP--21-278ea5" alt="venue"/> <img src="https://img.shields.io/badge/Status-Accepted-success" alt="status"/> <img src="https://img.shields.io/badge/Last%20Updated-2021-2D333B" alt="update"/>

## Environment
+ pytorch==1.3.0
+ torch_geometric==1.3.2

## Train steps

### Pre-train

+ `python run.py --pre_train True`

### Train policy reasoning

+ `python run.py --train_reasoning True --learningrate 1e-4 --epoch 20`

### Train recommendation module

+ `python run.py --train_rec True --learningrate 4e-4`

### Train dialog module

+ `python run.py --is_finetune True`
