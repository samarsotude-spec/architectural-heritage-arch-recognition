# Architectural Heritage Arch Recognition

Reproducible code and experimental resources for automatic arch recognition in architectural heritage images.

This repository accompanies a study on binary image-level classification of architectural heritage images into **Has Arch** and **No Arch** categories using transfer learning and conventional machine-learning baselines.

## Dataset

The complete curated dataset contains 7,479 images:

- Has Arch: 4,384
- No Arch: 3,095

A fixed stratified split was used throughout the experiments:

- Training: 5,235 images
- Validation: 748 images
- Test: 1,496 images

The dataset metadata and exact experimental partitions are deposited separately in Mendeley Data. Original third-party image files are not redistributed in this repository.

## Models evaluated

- ResNet50
- DenseNet121
- MobileNetV2
- HOG + Linear SVM

## Repository status

This repository is currently being prepared for the public release associated with the research article.
