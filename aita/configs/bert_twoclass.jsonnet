local train_path = "/home/johnkeszler/harvard/leddit/data/aita-train.pkl";
local val_path = "/home/johnkeszler/harvard/leddit/data/aita-dev.pkl";
local test_path = "/home/johnkeszler/harvard/leddit/data/aita-test.pkl";

local transformer_model = "bert-base-uncased";
local transformer_dim = 768;
local cls_is_last_token = false;
local batch_size = 4;
local max_seq_length = 440;
local epochs = 5;
local dropout = 0.1;
local lr = 1e-5;
local max_training_records = 5000;

{
  "dataset_reader":{
    "type": "aita_transformer_reader",
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": transformer_model,
      "max_length": max_seq_length
    },
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": transformer_model,
        "max_length": max_seq_length
      }
    },
	"two_classes": true,
	"remove_deleted": true,
  },


  "train_data_path": train_path,
  "validation_data_path": val_path,
  "test_data_path": test_path,

  "model": {
    "type": "basic_classifier",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "pretrained_transformer",
          "model_name": transformer_model,
          "max_length": max_seq_length
        }
      }
    },
    "seq2vec_encoder": {
       "type": "cls_pooler",
       "embedding_dim": transformer_dim,
       "cls_is_last_token": cls_is_last_token
    },
    "feedforward": {
      "input_dim": transformer_dim,
      "num_layers": 2,
      "hidden_dims": [transformer_dim, 200],
      "activations": "tanh"
    },
    "dropout": dropout
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size": batch_size
    }
  },
  "trainer": {
    "num_epochs": epochs,
    "cuda_device" : 0,
    "validation_metric": "+accuracy",
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "cut_frac": 0.06
    },
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": lr,
      "weight_decay": 0.1,
    }
  }
}
