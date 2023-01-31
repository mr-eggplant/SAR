# 🌠 Towards Stable Test-Time Adaptation in Dynamic Wild World

This is the official project repository for [Towards Stable Test-Time Adaptation in Dynamic Wild World 🔗](https://openreview.net/pdf?id=g2YraF75Tj) by
Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Zhiquan Wen, Yaofo Chen, Peilin Zhao and Mingkui Tan **(ICLR 2023 Oral, Notable-Top-5%)**.

- 1️⃣ SAR conducts model learning at test time to adapt a pre-trained model to test data that has distributional shifts ☀️ 🌧 ❄️, such as corruptions, simulation-to-real discrepancies, and other differences between training and testing data. 
- 2️⃣ SAR aims to adapt a model in dymamic wild world, i.e., the test data stream may have mixed domain shifts, small batch size, and online imbalanced label distribution shifts (as shown in the figure below).

<p align="center">
<img src="figures/wild_settings.png" alt="wild_settings" width="100%" align=center />
</p>

Method: Sharpness-aware and reliable entropy minimization (SAR)
- 1️⃣ SAR conducts selective entropy minimization by excluding partial samples with noisy gradients out of adaptation.

- 2️⃣ SAR optimizes both entropy and the sharpness of entropy surface simutaneously, so that the model update is robust to those remaining samples with noisy gradients.


**Installation**:

EATA depends on

- Python 3
- [PyTorch](https://pytorch.org/) >= 1.0
- [timm](https://github.com/rwightman/pytorch-image-models)


**Data preparation**:

This repository contains code for evaluation on [ImageNet-C 🔗](https://arxiv.org/abs/1903.12261) with ResNet-50 and VitBase. But feel free to use your own data and models!

- Step 1: Download [ImageNet-C 🔗](https://github.com/hendrycks/robustness) dataset from [here 🔗](https://zenodo.org/record/2235448#.YpCSLxNBxAc). 

- Step 2: Put IamgeNet-C at "--data_corruption" 

- Step 3 [optional, for EATA]: Put ImageNet **test/val set**  at  "--data".



**Usage**:

```
import sar
from sam import SAM

model = TODO_model()

model = sar.configure_model(model)
params, param_names = sar.collect_params(model)
base_optimizer = torch.optim.SGD
optimizer = SAM(params, base_optimizer, lr=args.lr, momentum=0.9)
adapt_model = sar.SAR(net, optimizer, margin_e0=0.4*math.log(1000))

outputs = adapt_model(inputs)  # now it infers and adapts!
```


## Example: Adapting a pre-trained model on ImageNet-C (Corruption).

**Usage**:

```
python3 main.py --data_corruption /path/to/imagenet-c --exp_type [normal/bs1/mix_shifts/label_shifts] --method [no_adapt/tent/eata/sar] --model [resnet50_gn_timm/vitbase_timm] --output /output/dir
```

'--exp_type' is choosen from:

- 'normal' means the same test setting to prior mild data stream in Tent and EATA

- 'bs1' means single sample adaptation, only one sample comes each time-step

- 'mix_shifts' conducts exps over the mixture of 15 corruption types in ImageNet-C

- 'label_shifts' means exps under online imbalanced label distribution shifts. Moreover, imbalance_ratio indicates the imbalance extent

Note: For EATA method, you need also to set "--data /path/to/imagenet" of clean ImageNet test/validation set to compute the weight importance for regularization.

**Results**:

Please see our [PAPER 🔗](https://openreview.net/pdf?id=g2YraF75Tj) for detailed results.



## Correspondence 

Please contact Shuaicheng Niu by niushuaicheng [at] gmail.com. 📬 


## Citation
If our SAR method or wild test-time adaptation settings are helpful in your research, please consider citing our paper:
```
@inproceedings{niu2023towards,
  title={Towards Stable Test-Time Adaptation in Dynamic Wild World},
  author={Niu, Shuaicheng and Wu, Jiaxiang and Zhang, Yifan and Wen, Zhiquan and Chen, Yaofo and Zhao, Peilin and Tan, Mingkui},
  booktitle = {Internetional Conference on Learning Representations},
  year = {2023}
}
```

## Acknowledgment
The code is greatly inspired by the [Tent 🔗](https://github.com/DequanWang/tent) and [EATA 🔗](https://github.com/mr-eggplant/EATA).
