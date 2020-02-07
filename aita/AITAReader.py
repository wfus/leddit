"""AllenNLP readers for my own AITA Dataset"""
from typing import List, Sequence, Iterable, Tuple, Dict
import itertools
import json
import logging
import pandas as pd

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

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

        fields = {
            'tokens': TextField(tokens, self._token_indexers),
            'label': LabelField(row.label),
        }
        return Instance(fields)
