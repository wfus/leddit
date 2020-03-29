"""AllenNLP readers for my own AITA Dataset"""
from typing import List, Sequence, Iterable, Tuple, Dict
import itertools
import json
import logging
import pandas as pd
import numpy as np

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from sklearn.utils import resample

logger = logging.getLogger(__name__)


@DatasetReader.register("aita_simple_reader")
class AITASimpleDataset(DatasetReader):
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
    def __init__(self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        lazy: bool = False,
        two_classes: bool = False,
        max_samples: int = -1,
        remove_deleted: bool = False,
        only_title: bool = False,
        ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.max_samples = max_samples
        self.two_classes = two_classes
        self.remove_deleted = remove_deleted
        self.only_title = only_title
        self.TWO_CLASSES = ['NTA', 'YTA']


    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        df = pd.read_pickle(file_path)
        logger.info("Label Initial Counts")
        logger.info(df.label.value_counts())

        # Check if we are only using two classes for training.
        if self.two_classes:
            logger.info("Simplifying dataset to only use 2 classes")
            logger.info("Using classes: %s", self.TWO_CLASSES)
            df = df[df.label.isin(self.TWO_CLASSES)]
        
        # Make sure that we don't have any empty titles
        df['title_stripped'] = df.title.map(lambda x: x.strip())
        df['title_stripped_len'] = df.title_stripped.map(len)
        df = df[df.title_stripped_len > 0]

        if not self.only_title:
            # Remove all deleted posts if specified
            if self.remove_deleted:
                logger.info("Using classes: %s", self.TWO_CLASSES)
                df = df[df.selftext.map(lambda x: "[removed]" not in x)]
                logger.info("New Label Counts without [removed]:")
                logger.info(df.label.value_counts())
                df = df[df.selftext.map(lambda x: "[deleted]" not in x)]
                logger.info("New Label Counts without [deleted]:")
                logger.info(df.label.value_counts())
        else:
            df.selftext = ""


        logger.info("Resampling labels, since resample_labels was set to true.")
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

        if self.max_samples > 0 and self.max_samples < len(df):
            df = df.sample(self.max_samples)

        for _, row in df.iterrows():
            yield self.text_to_instance(row.title, row.selftext, row.label)
    

    def text_to_instance(self, 
        title: str,
        post: str,
        label: str = None) -> Instance:
        """Converts a dataframe row into an instance."""
        post_tokens = [Token(word) for word in post]
        if len(post_tokens) == 0:
            post_tokens = [Token("empty")]
        title_tokens = [Token(word) for word in title]
        if len(title_tokens) == 0:
            title_tokens = [Token("empty")]

        post_field = TextField(post_tokens, self._token_indexers)
        title_field = TextField(title_tokens, self._token_indexers)

        fields = {
            'post': post_field,
            'title': title_field,
        }

        if label:
            fields["label"] = LabelField(label)

        return Instance(fields)


@DatasetReader.register("aita_transformer_reader")
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
        two_classes: bool = False,
        max_samples: int = -1,
        remove_deleted: bool = False,
        only_title: bool = False,
        **kwargs,
    ) -> None:
        """Dataset readers for using transformer-based downstream classification
        networks. Tested for BERT and Roberta.
            tokenizer: Type of tokenizer. Usually 'pretrained-transformer'.
            token_indexers: Token indexer. Same params at tokenizer usually.
            combine_input_fields: don't need to regard this.
            two_classes: Use only YTA or NTA rather than ESH and NAH.
            max_samples: Limit the number of samples in each epoch by sampling.
            remove_deleted: should we remove posts with [deleted] or [removed]
                should be false if we only use the title for predictions
            only_title: Should we only have the titles for prediction.
        """
        super().__init__(**kwargs)
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_samples = max_samples
        self.two_classes = two_classes
        self.remove_deleted = remove_deleted
        self.only_title = only_title
        if combine_input_fields is not None:
            self._combine_input_fields = combine_input_fields
        else:
            self._combine_input_fields = isinstance(self._tokenizer, PretrainedTransformerTokenizer)

        self.TWO_CLASSES = ['NTA', 'YTA']

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        df = pd.read_pickle(file_path)
        logger.info("Label Initial Counts")
        logger.info(df.label.value_counts())

        # Check if we are only using two classes for training.
        if self.two_classes:
            logger.info("Simplifying dataset to only use 2 classes")
            logger.info("Using classes: %s", self.TWO_CLASSES)
            df = df[df.label.isin(self.TWO_CLASSES)]


        if not self.only_title:
            # Remove all deleted posts if specified
            if self.remove_deleted:
                logger.info("Using classes: %s", self.TWO_CLASSES)
                df = df[df.selftext.map(lambda x: "[removed]" not in x)]
                logger.info("New Label Counts without [removed]:")
                logger.info(df.label.value_counts())
                df = df[df.selftext.map(lambda x: "[deleted]" not in x)]
                logger.info("New Label Counts without [deleted]:")
                logger.info(df.label.value_counts())
        else:
            df.selftext = ""


        logger.info("Resampling labels, since resample_labels was set to true.")
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

        if self.max_samples > 0 and self.max_samples < len(df):
            df = df.sample(self.max_samples)

        for _, row in df.iterrows():
            yield self.text_to_instance(row.title, row.selftext, row.label)

    @overrides
    def text_to_instance(
        self,  # type: ignore
        title: str,
        post: str,
        label: str = None,
    ) -> Instance:

        fields = {}
        tokens = self._tokenizer.tokenize_sentence_pair(title, post)
        fields["tokens"] = TextField(tokens, self._token_indexers)

        if label:
            fields["label"] = LabelField(label)

        return Instance(fields)
