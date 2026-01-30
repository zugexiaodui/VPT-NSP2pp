# VPT-NSP2++: Importance-Aware Visual Prompt Tuning in Null Space for Continual Learning

## Environment

- GPU: NVIDIA GeForce RTX 4090
- Python: 3.11.5

```
torch==2.1.0
torchvision==0.16.0
timm==0.9.12
einops==0.7.0
ftfy==6.1.3
huggingface-hub==0.18.0
numpy==1.26.0
opencv-python==4.8.1.78
Pillow==10.0.1
regex==2023.12.25
scikit-image==0.22.0
scikit-learn==1.3.2
scipy==1.11.3
tqdm==4.66.1
```
These packages can be installed easily by
`pip install -r requirements.txt`

## Dataset preparation
### 1. Download the datasets and uncompress them:

- CIFAR-100: https://www.cs.toronto.edu/~kriz/cifar.html
- ImageNet-R: https://github.com/hendrycks/imagenet-r
- DomainNet: https://ai.bu.edu/M3SDA/
- RESISC: https://www.kaggle.com/datasets/aqibrehmanpirzada/nwpuresisc45/data
- EuroSAT: https://github.com/phelber/EuroSAT
- OmniBenchmark: https://github.com/ZhangYuanhan-AI/OmniBenchmark

### 2. Rearrange the directory structure:

We use a unified directory structure for all datasets:
```
DATA_ROOT
    |- train
    |    |- class_folder_1
    |    |    |- image_file_1
    |    |    |- image_file_2
    |    |- class_folder_2
    |         |- image_file_2
    |         |- image_file_3
    |- val
         |- class_folder_1
         |    |- image_file_5
         |    |- image_file_6
         |- class_folder_2
              |- image_file_7
              |- image_file_8
```
We provide the scripts `split_[dataset].py` in the `tools` folder to rearange the directory structure.
Please change the `root_dir` in each script to the path of the uncompressed dataset.

## Training and evaluation

To start training and evaluation, use:

`bash train_eval.sh -t {number of tasks} -d {dataset name ('cifar100', 'imagenet_r', 'sdomainet', 'eurosat', 'resisc', 'omni')} --data_root {data folder containing 'train' and 'val'}`

Please specify the `--data_root` argument in the above bash script to the location of the datasets.
Change the `--seed` argument to use different seeds (e.g., 2026, 2027).

## Citation
```
(The BibTeX entry will be updated after publication)
@ARTICLE{11296947,
  author={Zhang, Shizhou and Lu, Yue and Cheng, De and Xing, Yinghui and Wang, Nannan and Wang, Peng and Zhang, Yanning},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={VPT-NSP2++: Importance-Aware Visual Prompt Tuning in Null Space for Continual Learning}, 
  year={2025},
  volume={},
  number={},
  pages={1-18},
  doi={10.1109/TPAMI.2025.3642298}}
```
