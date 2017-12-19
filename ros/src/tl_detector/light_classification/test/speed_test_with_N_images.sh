#!/usr/bin/env bash

IMAGE=out_300022.jpg
GRAPH=../model/retrained_graph.pb
LABELS=../model/retrained_labels.txt

python -m  label_N_images --image="$IMAGE" --graph="$GRAPH" --labels="$LABELS"

#optional arguments:
#  -h, --help            show this help message and exit
#  --image IMAGE         image to be processed
#  --graph GRAPH         graph/notes to be executed
#  --labels LABELS       name of file containing labels
#  --input_height INPUT_HEIGHT
#                        input height
#  --input_width INPUT_WIDTH
#                        input width
#  --input_mean INPUT_MEAN
#                        input mean
#  --input_std INPUT_STD
#                        input std
#  --input_layer INPUT_LAYER
#                        name of input layer
#  --output_layer OUTPUT_LAYER
#                        name of output layer
