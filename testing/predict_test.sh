#!/bin/sh

allennlp predict \
--include-package aita.AITAPredictor \
--include-package aita.test_reader \
--predictor aita_predictor \
$1 data/aita_prediction_example.jsonl
