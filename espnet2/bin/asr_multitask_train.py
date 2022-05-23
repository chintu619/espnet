#!/usr/bin/env python3
from espnet2.tasks.asr_multitask import ASRTask


def get_parser():
    parser = ASRTask.get_parser()
    return parser


def main(cmd=None):
    r"""ASR training.

    Example:

        % python asr_multitask_train.py asr --print_config --optim adadelta \
                > conf/train_asr.yaml
        % python asr_multitask_train.py --config conf/train_asr_multitask.yaml
    """
    ASRTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
