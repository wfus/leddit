local train_path = "/home/wfu/harvard/leddit/data/aita-train.pkl";
local val_path = "/home/wfu/harvard/leddit/data/aita-dev.pkl";
local test_path = "/home/wfu/harvard/leddit/data/aita-test.pkl";

local transformer_model = "roberta-base";
local transformer_dim = 768;
local cls_is_last_token = false;

{
  "dataset_reader":{
    "type": "aita_test_reader",
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": transformer_model
    },
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": transformer_model,
        "max_length": 512
      }
    }
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
          "max_length": 512
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
      "num_layers": 1,
      "hidden_dims": transformer_dim,
      "activations": "tanh"
    },
    "dropout": 0.1
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size": 1 
    }
  },
  "trainer": {
    "num_epochs": 10,
    "cuda_device" : 0,
    "validation_metric": "+accuracy",
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "cut_frac": 0.06
    },
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 2e-5,
      "weight_decay": 0.1,
    }
  }
}
