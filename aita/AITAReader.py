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
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from sklearn.utils import resample

logger = logging.getLogger(__name__)


@DatasetReader.register("aita_simple_reader")
class AITASimpleDataset(DatasetReader):
    """Rates posts into three different categories based off of simple majority
    reddit votes. The three categories are:
        NAH: Not an asshole
        YTA: You're the asshole
        ESH: Everyone sucks here
    """
    # The data files are actually as a pickled dataframe.
    def __init__(self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        lazy: bool = False) -> None:
        super().__init__()
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        """Load in a pickle with our dataframe. Unfortunately, it's pickled, so
        we cannot load it in lazily, so returns a list."""
        file_path = cached_path(file_path)
        df = pd.read_pickle(file_path)
        return list(df.apply(self.row_to_instance, axis=1))
    
    def row_to_instance(self, row):
        """Converts a dataframe row into an instance."""
        post = row.selftext
        title = row.title
        label = row.label
        post_tokens = [Token(word) for word in post]
        title_tokens = [Token(word) for word in title]

        post_field = TextField(post_tokens, self._token_indexers)
        title_field = TextField(title_tokens, self._token_indexers)
        label_field = LabelField(label)

        fields = {
            'post': post_field,
            'title': title_field,
            'label': label_field
        }
        return Instance(fields)

@DatasetReader.register("aita_bert_simple_reader")
class AITASimpleOnelineDataset(DatasetReader):
    """Rates posts into three different categories based off of simple majority
    reddit votes. The three categories are:
        NAH: noones an asshole here
        YTA: You're the asshole
        ESH: Everyone sucks here
    """
    # The data files are actually as a pickled dataframe.
    def __init__(self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        lazy: bool = False,
        resample_labels: bool = False,
        only_title: bool = False) -> None:
        super().__init__()
        """
        lazy: whether or not we should read line by line rather than read
            everything in all at once
        only_title: only train with the titles of the asshole posts
        resample_labels: resample the labels to equal proportion.
        """
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.only_title = only_title
        self.resample_labels = resample_labels

    @overrides
    def _read(self, file_path):
        """Load in a pickle with our dataframe. Unfortunately, it's pickled, so
        we cannot load it in lazily, so returns a list."""
        file_path = cached_path(file_path)
        df = pd.read_pickle(file_path)
        logger.info("Label Initial Counts")
        logger.info(df.label.value_counts())

        if self.resample_labels:
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
        
        return list(df.apply(self.row_to_instance, axis=1))
    
    def row_to_instance(self, row):
        """Converts a dataframe row into an instance."""
        post = row.selftext
        title = row.title
        if self.only_title:
            fullpost = title
        else:
            fullpost = title + post
        tokens = self._tokenizer.tokenize(fullpost)

        fields = {
            'tokens': TextField(tokens, self._token_indexers),
            'label': LabelField(row.label),
        }
        return Instance(fields)

@DatasetReader.register("aita_bert_fine_grained_reader")
class AITAFineGrainedOnelineDataset(DatasetReader):
    """Rates posts into three different categories based off of simple majority
    reddit votes. The three categories are:
        NAH: Not an asshole
        YTA: You're the asshole
        ESH: Everyone sucks here
    """
    # The data files are actually as a pickled dataframe.
    def __init__(self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        lazy: bool = False,
        only_title: bool = False) -> None:
        super().__init__()
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.only_title = only_title

    @overrides
    def _read(self, file_path):
        """Load in a pickle with our dataframe. Unfortunately, it's pickled, so
        we cannot load it in lazily, so returns a list."""
        file_path = cached_path(file_path)
        df = pd.read_pickle(file_path)
        return list(df.apply(self.row_to_instance, axis=1))
    
    def row_to_instance(self, row):
        """Converts a dataframe row into an instance."""
        post = row.selftext
        title = row.title
        if self.only_title:
            fullpost = title
        else:
            fullpost = title + post
        tokens = self._tokenizer.tokenize(fullpost)

        # Calculate probabilites of this row being various labels
        total = sum(row.label_fine.values())
        probs = []
        for key in sorted(row.label_fine):
            probs.append(row.label_fine[key]/total)

        fields = {
            'tokens': TextField(tokens, self._token_indexers),
            'label': ArrayField(np.array(probs)),
        }
        return Instance(fields)
