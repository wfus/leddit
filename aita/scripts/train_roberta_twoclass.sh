#!/bin/sh
# Run this from the base directory
export PYTHONPATH=.
allennlp train testing/test_roberta_custom.jsonnet \
--include-package aita.AITAReader \
-s $1 \
-f
