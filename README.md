# Transformer Module Networks (TMNs)

Repository for "Transformer Module Networks (TMNs)".   

This repository is the official PyTorch implementation of the paper ["Transformer Module Networks for Systematic Generalization in Visual Question Answering
"](https://arxiv.org/abs/2201.11316) by Moyuru Yamada, Vanessa D'Amario, Kentaro Takemoto, Xavier Boix, and Tomotake Sasaki. 

Provide the codes to train and test the following methods:  
- Standard Transformer (`base_model_*`)
- Standard Transformer with programs (`base_model_pg_*`)
- TMN - Stack architecture (`module_*_s`)
- TMN - Tree architecture (`module_*_t`)

I used a famous implementation of Transformer by [Hugging Face](<https://github.com/huggingface/transformers>).  

## Environment

Singularity 3.x

Base image: [nvcr.io/nvidia/pytorch:20.12-py3](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_20-12.html#rel_20-12)  

Additional libraries:  
[numpy](https://pypi.python.org/pypi/numpy) [tqdm](https://pypi.python.org/pypi/tqdm) [pillow](https://pypi.org/project/Pillow/) **[transformers](https://pypi.python.org/pypi/transformers)==2.11.0**  
(Some of them are already installed in the image)

## Data Setup

Paths to the data are written in [`path_cfgs.py`](./cfgs/path_cfgs.py).

### CLEVR, CoGenT, and CLOSURE

Download the original dataset ([CLEVR, CoGenT](https://cs.stanford.edu/people/jcjohns/clevr/), and [CLOSURE](https://github.com/rizar/CLOSURE)).  
- Grid features  
We simply use ResNet-101 as a feature extractor. Specifically, `torchvision.models.resnet101(pretrained=True)` . Output of the layer4 is W x H x 2048 dimensions. We flatten the W x H to N tokens (each visual token is 2048 dimentions).

- Regional features w/o pre-training  
We follow the approach in ["Neuro-Symbolic Concept Learner"](https://github.com/vacancy/NSCL-PyTorch-Release). Only use predicted bounding boxes.

- Pre-trained Object detector as a feature extractor  
Follow the instruction [here](https://github.com/peteanderson80/bottom-up-attention#demo).   
This object detector is pre-trained with Visual Genome. We extracted constant 36 object features per image.

### GQA

To get object features for training and test, follow the instruction of LXMERT [here](https://github.com/airsplay/lxmert#gqa).  
We use `vg_gqa_obj36.tsv` for training and `gqa_testdev_obj36.tsv` for test.  Model is trained on the whole GQA corpus (train + validation).

Programs can be obtain from a repository of "Meta-Module Network" [here](https://github.com/wenhuchen/Meta-Module-Network#data-downloading). We use `trainval_balanced_inputs.json` for  training and `testdev_balanced_inputs.json` for test.

## GQA-SGL
GQA-SGL is a novel test set we built. There are 4 tests or question types (verify, query, choose, and logical). 50 questions with ground-truth program for each question types (total 200 samples). Each test comes with two json files (`*_inputs.json` for NMN, `*_base.json` for standard Transformer). All json files are in `./datasets/`.

verify : `test_choose_attr_base.json`, `test_choose_attr_inputs.json`  
query : `test_query_relate_base.json`, `test_query_relate_inputse.json`  
choose : `test_verify_attr_base.json`, `test_verify_attr_inputs.json`  
logical : `test_exist_and_base.json`, `test_exist_and_inputs.json`  

## CLEVR, CoGenT, and CLOSURE

`--vf *` :  `region` for regional features, `vt` for grid features.  
`--tgt *` :  `clevr` for CLEVR or CLOSURE, `cgt` for CoGenT.

### Standard Transformer (Baseline)


#### w/ Grid features or Regional features
Training 
```bash
python3 train_clevr_base_vf.py --save_name *** --vf vt --tgt clevr
```

Test on CoGenT validation condition B
```bash
python3 eval_clevr_base_vf.py --from_pretrained ***/***.bin --vf vt --test valB
```

Test on CLOSURE
```bash
python3 eval_closure_base_vf.py --from_pretrained ***/***.bin
```

#### w/ regional features extracted by [a pre-trained object detector](https://github.com/peteanderson80/bottom-up-attention)
Training  
```bash
python3 train_clevr_base.py --save_name ***
```

Test
```bash
python3 eval_clevr_base.py --from_pretrained ***/***.bin
```

### Standard Transformer with ground-truth program (Baseline)

#### w/ Grid features or Regional features
Training  
```bash
python3 train_clevr_base_pg_vf.py --save_name *** --vf vt --tgt cgt
```

Test on CoGenT validation condition B
```bash
python3 eval_clevr_base_pg_vf.py --from_pretrained ***/***.bin --vf vt --test valB
```

Test on CLOSURE
```bash
python3 eval_closure_base_pg_vf.py --from_pretrained ***/***.bin
```

### Transformer Module Network
`--arch` : select network architecture (`s` for stack, `t` for tree)
`--vf *` : `region` for regional features, `vt` for grid features.
`--tgt *` :  `clevr` for CLEVR or CLOSURE, `cgt` for CoGenT.

#### w/ Grid features or Regional features

Training 
```bash
python3 train_clevr_program_vf.py --save_name *** --arch s --vf vt --tgt cgt
```

Test on CoGenT validation condition B
```bash
python3 eval_clevr_program_vf.py --from_pretrained ***/***.bin --arch s --vf vt --test valB
```

Test on CLOSURE
```bash
python3 eval_closure_program_vf.py --from_pretrained ***/***.bin --arch s --vf vt
```

#### w/ regional features extracted by [a pre-trained object detector](https://github.com/peteanderson80/bottom-up-attention)
Training 
```bash
python3 train_clevr_program.py --save_name *** --arch s
```

Test
```bash
python3 eval_clevr_program.py --from_pretrained ***/***.bin --arch s
```

# GQA
### Standard Transformer (Baseline)

Training 
```bash
python3 train_gqa_base.py --save_name ***
```

Test on GQA test-dev
```bash
python3 eval_gqa_base.py --from_pretrained ***/***.bin
```

Test on GQA-SGL
```bash
python3 eval_gqa_base.py --from_pretrained ***/***.bin --test ood
```

### Standard Transformer with ground-truth program (Baseline)

Training  
```bash
python3 train_gqa_base_pg.py --save_name ***
```

Test on GQA test-dev
```bash
python3 eval_gqa_base_pg.py --from_pretrained ***/***.bin
```

Test on GQA-SGL
```bash
python3 eval_gqa_base_pg.py --from_pretrained ***/***.bin --test ood
```

### Transformer Module Network
`--arch` : select network architecture (`s` for stack, `t` for tree)

Training 
```bash
python3 train_gqa_program.py --save_name *** --arch s
```

Test on GQA test-dev
```bash
python3 eval_gqa_program.py --from_pretrained ***/***.bin --arch s
```

Test on GQA-SGL
```bash
python3 eval_gqa_program.py --from_pretrained ***/***.bin --arch s --test ood
```

## Analysis

### Module specialization

`--func_map` : select a method for specializations of modules (`func` for semantic group, `random` for random group, and `order` for order)

Training
```bash
python3 train_clevr_program_sm.py --vf vt --func_map func
```

Test on CLOSURE
```bash
python3 eval_closure_program_sm.py --from_pretrained ***/***.bin --vf vt --func_map func
```

### Ablations
Set `--vl` and `--st` to enable 'Variable number of layers (VL)' and 'Split tokens (ST)'.  

Training
```bash
python3 train_clevr_ablation_vf.py --vf vt --vl --st
```

Test on CLOSURE
```bash
python3 eval_closure_ablation_vf.py --from_pretrained ***/***.bin --vf vt --vl --st
```

## License
This project is licensed under the terms of the BSD 3-Clause Clear License. Copyright 2022 Fujitsu Limited. All Rights Reserved.
