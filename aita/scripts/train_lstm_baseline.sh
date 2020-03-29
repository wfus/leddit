#!/bin/sh
# Run this from the base directory
export PYTHONPATH=.
allennlp train aita/configs/lstm_baseline.jsonnet \
--include-package aita.AITAReader \
--include-package aita.AITAModel \
-s $1 \
-f
