#!/usr/bin/env bash

IMAGE=images/sim_green.jpg
GRAPH=../model/mobilenet_1.0_224-2017_12_19_162810/optimized_graph.pb
LABELS=../model/mobilenet_1.0_224-2017_12_19_162810/retrained_labels.txt


python -m  label_N_images --image="$IMAGE" --graph="$GRAPH" --labels="$LABELS"

#Speed Test Results:
#mobilenet_1.0_224-2017_12_19_162810
#1.No optimized
#Total Evaluation time (100-images):      15.976s
#Average evaluation time (15.976/100):    0.160s
#2.No optimized
#Total Evaluation time (100-images):      14.720s
#Average evaluation time (14.720/100):    0.147s
#3.Rounded
#Total Evaluation time (100-images):      14.580s
#Average evaluation time (14.580/100):    0.146s



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
