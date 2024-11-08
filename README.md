# Deep-Citation
Code and experiments accompanying our paper **[Fine-Tuning Language Models on Multiple Datasets for Citation Intention Classification](https://arxiv.org/pdf/2410.13332)** at EMNLP 2024

## Requirement

Our code are developed using PyTorch. Please see ```KimNLP.yml``` for dependencies.

## How to run

To fine-tune a PLM for citation intention classification, please run

```
python main.py --dataset acl --lambdas 1 --data_dir Data/ --workspace Workspace/wksp --lm scibert
```

To jointly fine-tune a PLM on multiple datasets, run

```
python main.py --dataset acl-scicite-kim --lambdas 1-0.1-0.1 --data_dir Data/ --workspace Workspace/wksp --lm scibert
```

where datasets and the corresponding lambda values are concatenated using "-". The first dataset is the primary one while the others are the auxiliary datasets. Validation and Test F1 are evaluated on the primary dataset.

To compute the value of lambda using our TRL method, run

```
python trl.py --primary_dataset acl --auxiliary_dataset scicite --lm scibert --workspace Workspace/trl_wksp
```

## KIM Dataset

We are currently resolving the copyright issue of the KIM dataset. We will release it shortly.

## Cite
Please cite our paper if you find this repo useful:

```
@misc{shui2024deepcite,
      title={Fine-Tuning Language Models on Multiple Datasets for Citation Intention Classification}, 
      author={Zeren Shui and Petros Karypis and Daniel S. Karls and Mingjian Wen and Saurav Manchanda and Ellad B. Tadmor and George Karypis},
      year={2024},
      eprint={2410.13332},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.13332}, 
}
```