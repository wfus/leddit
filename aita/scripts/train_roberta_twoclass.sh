#!/bin/sh
# Run this from the base directory
export PYTHONPATH=.
allennlp train aita/configs/roberta_twoclass.jsonnet \
--include-package aita.AITAReader \
-s $1 \
-f
