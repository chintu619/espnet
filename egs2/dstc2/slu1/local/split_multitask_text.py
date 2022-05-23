#!/usr/bin/env python3

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('text_filepath', type=str, help="filepath to multitask text file")
    parser.add_argument('num_splits', type=int, help="number of splits made from text file")
    parser.add_argument('token_split', type=str, help="token used to split text from multiple tasks")
    return parser.parse_args()

def main(args):
    with open(args.text_filepath, 'r') as f:
        lines = [line.rstrip() for line in f.readlines()]
    files_write = []
    for i in range(args.num_splits):
        files_write.append(open(args.text_filepath+f"_task{i+1}", 'w'))
    for ix,line in enumerate(lines):
        key = line.split()[0]
        texts = ' '.join(line.split()[1:]).split(args.token_split)
        assert len(texts) == args.num_splits, print(f"{args.token_split} does not split the line:{ix+1} " + 
                                                    f"in text file into {args.num_splits} parts", line)
        for i in range(args.num_splits):
            text = texts.pop()
            files_write[i].write(f"{key} {' '.join(text.split())}\n")
    return

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
