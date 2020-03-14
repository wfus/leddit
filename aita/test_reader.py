from typing import Dict
import json
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer, PretrainedTransformerTokenizer

import pandas as pd
from sklearn.utils import resample

logger = logging.getLogger(__name__)


@DatasetReader.register("aita_test_reader")
class AITATestReader(DatasetReader):
    """
    Reads a file from the Stanford Natural Language Inference (SNLI) dataset.  This data is
    formatted as jsonl, one json-formatted instance per line.  The keys in the data are
    "gold_label", "sentence1", and "sentence2".  We convert these keys into fields named "label",
    "premise" and "hypothesis", along with a metadata field containing the tokenized strings of the
    premise and hypothesis.
    # Parameters
    tokenizer : `Tokenizer`, optional (default=`SpacyTokenizer()`)
        We use this `Tokenizer` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : `Dict[str, TokenIndexer]`, optional (default=`{"tokens": SingleIdTokenIndexer()}`)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    combine_input_fields : `bool`, optional
            (default=`isinstance(tokenizer, PretrainedTransformerTokenizer)`)
        If False, represent the premise and the hypothesis as separate fields in the instance.
        If True, tokenize them together using `tokenizer.tokenize_sentence_pair()`
        and provide a single `tokens` field in the instance.
    """

    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        combine_input_fields: bool = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        if combine_input_fields is not None:
            self._combine_input_fields = combine_input_fields
        else:
            self._combine_input_fields = isinstance(self._tokenizer, PretrainedTransformerTokenizer)

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        df = pd.read_pickle(file_path)
        logger.info("Label Initial Counts")
        logger.info(df.label.value_counts())

        logger.info("Resampling labels, since resample_labels"
            " was set to true.")
        labels = list(df.label.unique())
        label_dataframes = []
        for label in labels:
            label_dataframes.append(df[df.label == label])
        label_counts = [len(x) for x in label_dataframes]
        largest_label = max(label_counts)
        df = pd.concat([
            resample(label_df,
                replace=True,
                n_samples=largest_label,
                random_state=420)
            for label_df in label_dataframes])
        logger.info("New label sampling is:")
        logger.info(df.label.value_counts())

        for _, row in df.iterrows():
            yield self.text_to_instance(row.title, row.selftext, row.label)

    @overrides
    def text_to_instance(
        self,  # type: ignore
        title: str,
        post: str,
        label: str = None,
    ) -> Instance:

        fields: Dict[str, Field] = {}
        tokens = self._tokenizer.tokenize_sentence_pair(title, post)
        fields["tokens"] = TextField(tokens, self._token_indexers)

        if label:
            fields["label"] = LabelField(label)

        return Instance(fields)
