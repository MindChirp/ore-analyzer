# YOLOv11 (You Only Look Once) for categorizing ores within an image

This repo is part of an even bigger assignment. For the other two repos which the assignment consists of, please see these links:

- [DiamondFinder](https://github.com/SkinnyAG/DiamondFinder/tree/master)
- [DiamondFinderPlugin](https://github.com/SkinnyAG/DiamondFinderPlugin)

## So what is this?

This repository contains a trained YOLOv11 model, specifically conditioned for recognizing five different ore types:

- Deepslate redstone ore
- Deepslate diamond ore
- Deepslate iron ore
- Deepslate gold ore
- Iron ore

In addition to classifying the ores in an image, bounding boxes are drawn around them.

## Training

The model has been trained on an [augmentet dataset](https://universe.roboflow.com/oblig10/minecraft-ore-detection-20pzg-xejw6/dataset/2), consisting of roughly 3.1k images.
The training lasted for 10 epochs, and yielded an mAP of ~0.7.

## The old CNN model

Additionally, there is a folder containing an old CNN model, which was used as a testing ground for using CNNs to categorize ores. Its precision deteriorated significantly over larger distances, and was therefore scrapped.
