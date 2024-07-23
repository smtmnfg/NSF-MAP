# Repository for "NS-HyMAP: Neurosymbolic Multimodal Hybrid Fusion for Robust and Interpretable Anomaly Prediction in Assembly Pipelines" Paper
This repository contains curated datasets, implementation of methods experimented and introduced in the paper titled "NS-HyMAP: Neurosymbolic Multimodal Hybrid Fusion for Robust and Interpretable Anomaly Prediction in Assembly Pipelines".

Baselines

This folder includes the baseline models developed.
Three baseline models:

Autoencoder:
To run py .Baselines/autoencoder.py

Custom ViT:
To run py .Baselines/custom_vit.py

CNN:
To run py .Baselines/image_with_seg_cnn.py

Pretrained-CNN:
To run py .Baselines/image_with_segmentation_pretrainedcnn.py

Pretrained-ViT:
To run py .Baselines/image_with_segmentation_vit.py

Proposed Fusion Approach

This folder includes the models for the proposed approach.

Data Preprocessing

This folder contains the codes used to extract the analog and multimodal datasets which was created from the Future Factories(FF) Dataset (https://www.kaggle.com/datasets/ramyharik/ff-2023-12-12-analog-dataset)

The final preprocessed multimodal dataset available at: https://drive.google.com/drive/folders/1l7_Blmk_RrsLHqcW_4F7-ELtll_ZGyDv?usp=sharing


The final preprocessed analog dataset available at: https://drive.google.com/drive/folders/1v-UyJqlZMG68Mwd2Gict9E885MS023iR?usp=sharing
