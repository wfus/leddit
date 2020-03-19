#!/bin/sh

allennlp serve \
--predictor aita_predictor \
--archive-path $1 \
--include-package aita.AITAPredictor \
--include-package aita.AITAReader