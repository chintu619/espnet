#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="valid"
test_sets="test"

asr_config=conf/train_multitask_transformer.yaml
# asr_config=conf/train_multitask_wav2vec.yaml
inference_config=conf/decode_asr.yaml

./asr_multitask.sh \
  --lang en \
  --ngpu 1 \
  --num_decoders 2 \
  --use_lm false \
  --nbpe 5000 \
  --stage 0 \
  --stop_stage 10000 \
  --token_type word \
  --feats_type raw \
  --gpu_inference false \
  --max_wav_duration 30 \
  --feats_normalize utterance_mvn \
  --inference_nj 12 \
  --inference_asr_model valid.acc.ave.pth \
  --inference_config ${inference_config} \
  --asr_config "${asr_config}" \
  --train_set "${train_set}" \
  --valid_set "${valid_set}" \
  --test_sets "${test_sets}" "$@"
