# Uncertainty Quantification in DL Models for Cervical Cytology
###  [Paper Link](https://unified.baulab.info) || [Code](https://github.com/shubhamOjha1000/Uncertainty-Quantification-in-DL-Models-for-Cervical-Cytology)
- #### Authors</ins>: **Shubham Ojha & Aditya Narendra**
-  #### Venue & Year: Medical Imaging with Deep Learning (MIDL), 2024
## Abstract

Deep Learning (DL) has demonstrated significant promise in digital pathological applications both histopathology and cytopathology. However, the majority of these works primarily concentrate on evaluating the general performance of the models and overlook the crucial requirement for uncertainty which is necessary for real-world clinical application. In this study, we examine the change in predictive performance and the identification of mispredictions through the incorporation of uncertainty estimates for DL-based Cervical cancer classification. Specifically, we evaluate the efficacy of three methods—Monte Carlo(MC) Dropout, Ensemble Methods, and Test Time Augmentation(TTA) using three metrics: variance, entropy, and sample mean uncertainty. The results demonstrate that integrating uncertainty estimates improves the model's predictive capacity in high-confidence regions, while also serving as an indicator for the model's mispredictions in low-confidence regions.

## Dataset Guide

**Center for Recognition and Inspection of Cells (CRIC) Dataset** : [Download Link](https://database.cric.com.br/downloads)

### Extract Cells from the Patches 
- To extract a cell crop size of 100 x 100 centered on the nucleus from the patches.
```
python Cropping.py --dataset='path to the cell centre coordinates csv file' --img_dir='path to the cric img patches directory' --cell_img_dir='path to cell img directory'
```

### Train-Val-Test Split
- To split the dataset for training, validation and testing purposes
```
python Train_val_test_split.py --dataset='path to your csv label file' 
```

## Experimentation Guide

### Run Baseline model
- To train the baseline model having ResNet50 as a feature extractor
```
python main.py --num_classes=2 --num_epochs=100 --img_dir='path to the img directory' --model='CustomResNet'   --batch_size=256
```

### Run MC Dropout model
- To train the MC Dropout model
```
python main.py --num_classes=2 --num_epochs=100 --img_dir='path to the img directory' --model='MC_Dropout_model' --multi_head_loss='avg_across_all_heads_loss'  --batch_size=256
```

### Run Ensemble Methods model
- To train the Ensemble Methods model
```
python main.py --num_classes=2 --num_epochs=100 --img_dir='path to the img directory' --model='EnsembleModel_ResNet' --multi_head_loss='meta_individual_multi_head_loss'  --batch_size=256
```

### Run Test Time Augumentation (TTA) model 
- To run 50 augmentations for each image sample
```
python TTA_transformations.py --start_idx=1 --end_idx=51 --img_dir='path to the img directory' --dataset='path to the label file' --TTA_path='path for TTA director'
```
- To join each sample to create a tensor shape of (50, 3, 224, 224)
```
python join_TTA_tensors.py --start_idx=1 --end_idx=11535 --TTA_path='path for TTA director'
```
- To train the TTA model
```
python main.py --num_classes=2 --num_epochs=100 --img_dir='path to the joined tensor' --model='TTA_model' --multi_head_loss='avg_across_all_heads_loss'  --batch_size=32 --num_workers=4
```

## Results


<p align="center">
<img src="https://github.com/shubhamOjha1000/Uncertainty-Quantification-in-DL-Models-for-Cervical-Cytology/assets/72977734/8d7a8fca-9af9-49a6-a101-351822b723cc " width="900" height="230"><br>
<b>Distribution of Sample Predictions across Uncertainty Ranges for MC Dropout</b><br>
</p>

<p align="center">
<img src="https://github.com/shubhamOjha1000/Uncertainty-Quantification-in-DL-Models-for-Cervical-Cytology/assets/72977734/f8b9312c-e8f8-4261-b97f-d6622ab1a854 " width="900" height="230"><br>
<b>Distribution of Sample Predictions across Uncertainty Ranges for Ensemble Method</b><br>
</p>

<p align="center">
<img src="https://github.com/shubhamOjha1000/Uncertainty-Quantification-in-DL-Models-for-Cervical-Cytology/assets/72977734/0298f7e4-b0a0-4e20-80ed-dc325877fc23 " width="900" height="230"><br>
<b>Distribution of Sample Predictions across Uncertainty Ranges for Test Time Augmentation (TTA)</b><br>
</p>

