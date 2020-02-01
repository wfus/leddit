"""Simple models for the AITA Dataset."""
from typing import List, Sequence, Iterable, Tuple, Dict
import itertools
import json
import logging
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

from allennlp.models.model import Model
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data import Vocabulary
from allennlp.nn import InitializerApplicator
from allennlp.nn import RegularizerApplicator
from allennlp.nn import util

from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder

logger = logging.getLogger(__name__)

@Model.register("aita_simple_model")
class AITASimpleClassifier(Model):
    """Simple model for AITA dataset."""
    def __init__(self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        text_encoder: Seq2VecEncoder,
        classifier_feedforward: FeedForward,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None) -> None:

        self.text_field_embedder = text_field_embedder
        self.text_encoder = text_encoder
        self.classifier_feedforward = classifier_feedforward

        if text_field_embedder.get_output_dim() != text_encoder.get_input_dim():
            raise ConfigurationError("Dimension mismatch between text field"
                " embedder and text encoder.")
        
        self.metrics = {
            "accuracy": CategoricalAccuracy()
        }

        self.loss = torch.nn.CrossEntropyLoss()
        initializer(self)
    
    @overrides
    def forward(self,
        tokens: Dict[str, torch.LongTensor],
        label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        pass

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pass

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        pass





