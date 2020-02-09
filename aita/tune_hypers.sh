#!/bin/sh

allentune search \
--experiment-name fullpost_tuning \
--num-gpus 1 \
--gpus-per-trial 1 \
--search-space aita/search_space.json \
--num-samples 30 \
--base-config aita/aita_simple_bert.jsonnet \
--include-package aita.AITAReader \
--include-package aita.AITASimpleModel
