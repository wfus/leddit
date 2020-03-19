#!/bin/sh

allennlp predict \
--include-package aita.AITAPredictor \
--include-package aita.AITAReader \
--predictor aita_predictor \
$1 data/aita_recent.jsonl \
--output-file data/aita_recent_output.jsonl
