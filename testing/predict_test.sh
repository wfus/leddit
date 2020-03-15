#!/bin/sh

allennlp predict \
--include-package aita.AITAPredictor \
--predictor aita_predictor \
$1 data/aita_prediction_example.jsonl