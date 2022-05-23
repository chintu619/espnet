#!/usr/bin/env python3

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('text_filepath', type=str, help="filepath to multitask text file")
    return parser.parse_args()

def main(args):
    with open(args.text_filepath, 'r') as f:
        lines = [line.rstrip() for line in f.readlines()]
    with open(args.text_filepath, 'w') as f:
        for line in lines:
            key = line.split()[0]
            text = " ".join(line.split()[1:]).replace("-", " ")
            f.write(f"{key} " + " ".join(text.split()) + "\n")
    return

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
