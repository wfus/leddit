"""AllenNLP readers for my own AITA Dataset"""
from typing import List, Sequence, Iterable, Tuple, Dict
import itertools
import json
import logging

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
class AITASimpleDataset(object):
    """Rates posts into three different categories based off of simple majority
    reddit votes. The three categories are:
        NAH: Not an asshole
        YTA: You're the asshole
        ESH: Everyone sucks here
    """
