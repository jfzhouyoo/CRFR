# Code for CRFR
> The implementation of our paper accepted by EMNLP 2021: [**CRFR: Improving Conversational Recommender Systems via Flexible Fragments Reasoning on Knowledge Graphs**](https://aclanthology.org/2021.emnlp-main.355/)

> The code and data are developed based on KGSF. Due to time reasons, it has not been transformed into a standardized code framework, so it seems that the code is bad.

<img src="https://img.shields.io/badge/Venue-EMNLP--21-278ea5" alt="venue"/> <img src="https://img.shields.io/badge/Status-Accepted-success" alt="status"/> <img src="https://img.shields.io/badge/Last%20Updated-2021-2D333B" alt="update"/>

## Environment
+ `python==3.6.12`
+ `torch==1.3.0+cu100`
+ `torch_geometric==1.3.2`

## Train steps

### Pre-train

+ `python run.py --pre_train True`

### Train policy reasoning

+ `python run.py --train_reasoning True --learningrate 1e-4 --epoch 20`

### Train recommendation module

+ `python run.py --train_rec True --learningrate 4e-4`

### Train dialog module

+ `python run.py --is_finetune True`

## Citation
If you find our work useful for your research, please kindly cite our paper as follows:
```
@inproceedings{DBLP:conf/emnlp/ZhouWHH21,
  author       = {Jinfeng Zhou and
                  Bo Wang and
                  Ruifang He and
                  Yuexian Hou},
  editor       = {Marie{-}Francine Moens and
                  Xuanjing Huang and
                  Lucia Specia and
                  Scott Wen{-}tau Yih},
  title        = {{CRFR:} Improving Conversational Recommender Systems via Flexible
                  Fragments Reasoning on Knowledge Graphs},
  booktitle    = {Proceedings of the 2021 Conference on Empirical Methods in Natural
                  Language Processing, {EMNLP} 2021, Virtual Event / Punta Cana, Dominican
                  Republic, 7-11 November, 2021},
  pages        = {4324--4334},
  publisher    = {Association for Computational Linguistics},
  year         = {2021},
  url          = {https://doi.org/10.18653/v1/2021.emnlp-main.355},
  doi          = {10.18653/v1/2021.emnlp-main.355},
  timestamp    = {Thu, 16 Jun 2022 20:35:19 +0200},
  biburl       = {https://dblp.org/rec/conf/emnlp/ZhouWHH21.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
