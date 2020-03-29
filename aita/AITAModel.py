"""Simple models for the AITA Dataset."""
from typing import List, Sequence, Iterable, Tuple, Dict, Optional
import itertools
import json
import logging
import torch

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer
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
        regularizer: Optional[RegularizerApplicator] = None
    ) -> None:
        super().__init__(vocab)
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
        post: Dict[str, torch.LongTensor],
        title: Dict[str, torch.LongTensor],
        label: torch.LongTensor = None,
    ) -> Dict[str, torch.Tensor]:
        post_emb = self.text_field_embedder(post)
        title_emb = self.text_field_embedder(title)
        mask_post = util.get_text_field_mask(post)
        mask_title = util.get_text_field_mask(title)
        enc_post = self.text_encoder(post_emb, mask_post)
        enc_title = self.text_encoder(title_emb, mask_title)

        logits = self.classifier_feedforward(
            torch.cat([enc_post, enc_title], dim=1))
        output_dict = {"logits": logits}

        if label is not None:
            loss = self.loss(logits, label)
            for metric in self.metrics.values():
                metric(logits, label)
            output_dict["loss"] = loss
        
        return output_dict


    def decode(self, output_dict: Dict[str, torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
        """Returns MLE estimate for our classes."""
        class_probs = F.softmax(output_dict['logits'], dim=-1)
        output_dict['label'] = class_probs.cpu().numpy().argmax(dim=-1)
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset)
                for metric_name, metric in self.metrics.items()}


@Model.register("lstm_baseline")
class AITALSTMBaseline(Model):
    def __init__(self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        text_encoder: Seq2VecEncoder,
        classifier_feedforward: FeedForward,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None):
        super().__init__(vocab)
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
        post: Dict[str, torch.LongTensor],
        title: Dict[str, torch.LongTensor],
        label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        post_emb = self.text_field_embedder(post)
        title_emb = self.text_field_embedder(title)
        mask_post = util.get_text_field_mask(post)
        mask_title = util.get_text_field_mask(title)
        enc_post = self.text_encoder(post_emb, mask_post)
        enc_title = self.text_encoder(title_emb, mask_title)

        logits = self.classifier_feedforward(
            torch.cat([enc_post, enc_title], dim=1))
        output_dict = {"logits": logits}

        if label is not None:
            loss = self.loss(logits, label)
            for metric in self.metrics.values():
                metric(logits, label)
            output_dict["loss"] = loss
        
        return output_dict

    def decode(self, output_dict: Dict[str, torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
        """Returns MLE estimate for our classes."""
        class_probs = F.softmax(output_dict['logits'], dim=-1)
        output_dict['label'] = class_probs.cpu().numpy().argmax(dim=-1)
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset)
                for metric_name, metric in self.metrics.items()}
