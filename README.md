# Pose Classification

## Description

This is a small MLP to check that an we can classify poses into some action labels such as `walking`, `standing`, and `sitting`.

## Structure

There's three main directories:
- `data` stores datapreprocessing scripts and a dataloader 
- `model` stores a single file, `pose_classifier.py` which is an MLP with 2 hidden layers
- `datasets` is where the preprocessed datasets are stored -- this should one small train/val split
 
