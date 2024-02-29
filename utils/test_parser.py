# -*- coding: utf-8 -*-
"""
Parsers from command line for testing the MRI-SAM model.
"""

import argparse

parser = argparse.ArgumentParser(
    description="run inference on testing set based on MRI-SAM"
)
parser.add_argument(
    "-i",
    "--data_path",
    type=str,
    default="none",
    help="path to the data folder",
)
parser.add_argument(
    "-o",
    "--seg_path",
    type=str,
    default="none",
    help="path to the segmentation folder",
)
parser.add_argument(
    "--prompt",
    type=list,
    default="point",
    help="Choose a type of prompt from 'point', 'bbox', and 'text'",
)
parser.add_argument("--device", type=str, default="cuda:0", help="device")
parser.add_argument(
    "-chk",
    "--checkpoint",
    type=str,
    default="None",
    help="path to the trained model",
)