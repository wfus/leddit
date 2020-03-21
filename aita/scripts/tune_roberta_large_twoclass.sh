#!/bin/sh

allentune search \
--experiment-name roberta_large_twoclass_tuning \
--num-gpus 1 \
--gpus-per-trial 1 \
--search-space aita/configs/roberta_large_twoclass_search.json \
--base-config aita/configs/roberta_large_twoclass_tune.jsonnet \
--num-samples 30 \
--include-package aita.AITAReader \
--include-package aita.AITASimpleModel
