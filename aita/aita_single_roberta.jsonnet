local train_path = "/home/johnkeszler/Documents/leddit/aita/aita-train.pkl";
local val_path = "/home/johnkeszler/Documents/leddit/aita/aita-dev.pkl";


local transformer_model = "roberta-base";
local transformer_size = 768;
local batch_size = 32;
local learning_rate = 2e-5;
local dropout = 0.1;

{
    "train_data_path": train_path,
    "validation_data_path": val_path,

    "dataset_reader": {
        "type": "aita_bert_simple_reader",
        "tokenizer_name": transformer_model,
        "max_seq_len": 256,
        "lazy": false,
        "only_title": false,
        "resample_labels": true,
    },

    "trainer": {
        "num_epochs": 5,
        "cuda_device": 0,
        "validation_metric": "+accuracy",
		"learning_rate_scheduler": {
			"type": "slanted_triangular",
			"cut_frac": 0.06,
		},
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": learning_rate,
            "weight_decay": 0.1,
        },
        "num_serialized_models_to_keep": 1,
        "grad_norm": 1.0,
    },

    "model": {
        "type": "basic_classifier",
		"text_field_embedder": {
			"token_embedders": {
				"tokens": {
					"type": "pretrained_transformer",
					"model_name": transformer_model,
					"max_length": 256,
				}
			}
		},
		"seq2vec_encoder": {
			"type": "cls_pooler",
			"embedding_dim": transformer_size,
            "cls_is_last_token": false,
		},
        "feedforward": {
            "input_dim": transformer_size,
            "num_layers": 2,
            "hidden_dims": [transformer_size, 4],
            "activations": "tanh"
        },
        "dropout": dropout,
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": batch_size
        }
    }
}
