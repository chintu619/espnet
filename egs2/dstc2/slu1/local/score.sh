#!/usr/bin/env bash
# Copyright 2021  Siddhant Arora
#           2021  Carnegie Mellon University

# # begin configuration section.
# cmd=run.pl
# stage=0
# data=data/eval2000
# #end configuration section.

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -lt 1 ]; then
  echo "Usage: local/score.sh <asr-inference-dir>"
  exit 1;
fi

asr_inferencedir=$1
python local/score.py ${asr_inferencedir}

_scoredir="${asr_inferencedir}/valid/score_wer/"
sclite \
            -r "${_scoredir}/ref_asr.trn" trn \
            -h "${_scoredir}/hyp_asr.trn" trn \
            -i rm -o all stdout > "${_scoredir}/result_asr.txt"
echo "Write ASR result in ${_scoredir}/result_asr.txt"
                grep -e Avg -e SPKR -m 2 "${_scoredir}/result_asr.txt"
_scoredir="${asr_inferencedir}/test/score_wer/"
sclite \
            -r "${_scoredir}/ref_asr.trn" trn \
            -h "${_scoredir}/hyp_asr.trn" trn \
            -i rm -o all stdout > "${_scoredir}/result_asr.txt"
echo "Write ASR result in ${_scoredir}/result_asr.txt"
                grep -e Avg -e SPKR -m 2 "${_scoredir}/result_asr.txt"
exit 0

