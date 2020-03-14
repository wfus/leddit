#!/bin/sh
# Run this from the base directory
export PYTHONPATH=.
allennlp train testing/test_roberta.jsonnet \
-s $1 \
-f
