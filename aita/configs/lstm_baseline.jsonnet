local train_path = "/home/wfu/harvard/leddit/data/aita-train.pkl";
local val_path = "/home/wfu/harvard/leddit/data/aita-dev.pkl";

{
  "dataset_reader":{
    "type": "aita_simple_reader",
    "tokenizer": "whitespace",
    "two_classes": true,
    "remove_deleted": true,
    "only_title": false,
  },

  "train_data_path": train_path,
  "validation_data_path": val_path,

  "model": {
    "type": "lstm_baseline",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
          "embedding_dim": 100,
          "trainable": false
        }
      }
    },
    "text_encoder": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 100,
      "hidden_size": 100,
      "num_layers": 2,
      "dropout": 0.2
    },
    "classifier_feedforward": {
      "input_dim": 400,
      "num_layers": 2,
      "hidden_dims": [200, 2],
      "activations": ["relu", "linear"],
      "dropout": [0.2, 0.0]
    }
  },

  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size": 16,
    },
  },
  "trainer": {
    "num_epochs": 40,
    "patience": 10,
    "cuda_device": 0,
    "grad_clipping": 5.0,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adagrad"
    }
  }
}
