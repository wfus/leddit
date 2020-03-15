"""Manual training script used for debugging before using configuration jsonnet
files with AllenNLP and tuning hyperparameters."""

from AITAReader import AITASimpleOnelineDataset
import numpy as np
from pathlib import Path



from typing import Iterator, List, Dict

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField

from allennlp.data.dataset_readers import DatasetReader

from allennlp.common.file_utils import cached_path

from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

from allennlp.data.vocabulary import Vocabulary

from allennlp.models import Model

from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits

from allennlp.training.metrics import CategoricalAccuracy

from allennlp.data.iterators import BucketIterator

from allennlp.training.trainer import Trainer

from allennlp.predictors import SentenceTaggerPredictor
from transformers import RobertaModel

TRAIN_FILE = Path.cwd().parent / 'data' / 'aita-tiny-train.pkl'
VAL_FILE = Path.cwd().parent / 'data' / 'aita-tiny-dev.pkl'


class TransformerModel(Model):
    """Quick transformer based model. We will use:
        * Bert
        * DistilBert
        * Roberta
    """
    def __init__(self, transformer_type):
        self.transformer = RobertaModel.from_pretrained(transformer_type)
        self.feedforward = nn.Linear(768, 4)
    
    def forward(self,
                tokens: Dict[str, torch.Tensor],
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        print(tokens)
        return


if __name__ == '__main__':
    reader = AITASimpleOnelineDataset(
        tokenizer_name='roberta-base',
        max_seq_len=512,
        lazy=False,
        resample_labels=True,
        only_title=False)
    train_dataset = reader.read(TRAIN_FILE)
    val_dataset = reader.read(VAL_FILE)
