# -*- coding: utf-8 -*-
"""
Parsers from command line for testing the MRI-SAM model.
"""

import argparse

# test parsers
test_parser = argparse.ArgumentParser(
    description="run inference on testing set based on MRI-SAM"
)
test_parser.add_argument(
    "-i",
    "--data_path",
    type=str,
    default="none",
    help="path to the data folder",
)
test_parser.add_argument(
    "-o",
    "--seg_path",
    type=str,
    default="none",
    help="path to the segmentation folder",
)
test_parser.add_argument(
    "--prompt",
    type=str,
    default="point",
    help="Choose a type of prompt from 'point', 'bbox', and 'text'",
)
test_parser.add_argument("--device", type=str, default="cuda:0", help="device")
test_parser.add_argument(
    "-e",
    "--encoder_tpye",
    type=str,
    default="vit_h",
    help="Choose a image encoder type from 'vit_h', 'vit_b', and 'vit-l'",
)
test_parser.add_argument(
    "-chk",
    "--checkpoint",
    type=str,
    default="None",
    help="path to the trained model",
)