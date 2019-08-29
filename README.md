# Creating Synthetic Image Datasets
This tool helps create synthetic data for object detection modeling. Given
a folder of background images and object images, this tool iterates through each
background and superimposes objects within the frame in random locations,
automatically annotating as it goes. The tool also resizes the icons to help the
model generalize better to the real world.

## Setup
Clone this repo. Then create and activate the conda environment provided:
```bash
$ conda env create -f environment.yml
$ conda activate images
```

Place background images in the `Backgrounds/` subfolder and objects in
the `Objects/` subfolder.

## Create
Run the `create.py` script to generate hundreds/thousands of synthetic training
images for object detection models.

```bash
$ python create.py
```

Output images will be placed in the `TrainingData/` subfolder once done.

### Args
These are the available entrypoint arguments that you can supply at runtime. More will be added in the future.

- `--backgrounds`: Path to folder of background images.
- `--objects`    : Path to folder of object images.
- `--output`     : Path to folder of output images.
- `--groups`     : Whether or not to place groups of objects together.
- `--annotate`   : Whether or not to create and save annotations for the new images.
- `--sframe`     : Whether or not to create a Turi Create SFrame for modeling.
