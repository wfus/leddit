#!/bin/sh
# Run this from the base directory
export PYTHONPATH=.
allennlp train aita/aita_simple.jsonnet \
--include-package aita.AITAReader \
--include-package aita.AITASimpleModel \
-s $1