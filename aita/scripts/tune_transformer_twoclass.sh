#!/bin/sh

allentune search \
--experiment-name transformer_title_twoclass \
--num-gpus 1 \
--gpus-per-trial 1 \
--search-space aita/configs/transformer_twoclass_search.json \
--base-config aita/configs/transformer_twoclass_tune.jsonnet \
--num-samples 30 \
--include-package aita.AITAReader \
--include-package aita.AITASimpleModel
