# CRFR

## Environment
+ pytorch==1.3.0
+ torch_geometric==1.3.2

> We will release the source code.

## Train steps

### Pre-train

+ `python run.py --pre_train True`

### Train policy reasoning

+ `python run.py --train_reasoning True --learningrate 1e-4 --epoch 20`

### Train recommendation module

+ `python run.py --train_rec True --learningrate 4e-4`

### Train dialog module

+ `python run.py --is_finetune True`