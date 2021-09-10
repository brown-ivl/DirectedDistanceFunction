
# Directed Distance Function

This repository contains all the code needed to train a network to learn the directed distance to an object surface. This includes code for data generation, model creation, and training.

## Data Generation / Sampling Rays

The data generation code offers a few methods for sampling rays with the idea that some sampling techniques produce harder to learn rays (edge cases). To visualize the different sampling methods, use

`python sampling.py -v --mesh_file <path to .obj>`

You can also see how fast each sampling method is by running

`python sampling.py -s`

## Training

To train, test, and save a network, run

`python train.py -Tts -n mynetwork --mesh_file <path to .obj>`

To see all flags use

`python train.py --help`
