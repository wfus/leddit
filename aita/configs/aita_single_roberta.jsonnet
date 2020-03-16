local train_path = "/home/wfu/harvard/leddit/aita/aita-tiny-train.pkl";
local val_path = "/home/wfu/harvard/leddit/aita/aita-tiny-dev.pkl";
local test_path = "/home/wfu/harvard/leddit/aita/aita-tiny-test.pkl";
# local train_path = "/home/wfu/harvard/leddit/aita/aita-small-train.pkl";
# local val_path = "/home/wfu/harvard/leddit/aita/aita-small-dev.pkl";
# local test_path = "/home/wfu/harvard/leddit/aita/aita-small-test.pkl";
# local train_path = "/home/wfu/harvard/leddit/aita/aita-train.pkl";
# local val_path = "/home/wfu/harvard/leddit/aita/aita-dev.pkl";
# local test_path = "/home/wfu/harvard/leddit/aita/aita-test.pkl";


local transformer_model = "roberta-large";
local transformer_size = 1024;
local cls_is_last_token = false;
local batch_size = 32;
local learning_rate = 2e-5;
local dropout = 0.1;

{
    "dataset_reader": {
        "type": "aita_bert_simple_reader",
        "tokenizer_name": transformer_model,
        "max_seq_len": 256,
        "lazy": false,
        "only_title": false,
        "resample_labels": true,
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
					"model_name": 'bert-base-uncased',
					"max_length": 256,
				}
			}
		},
		"seq2vec_encoder": {
			"type": "cls_pooler",
			"embedding_dim": transformer_size,
            "cls_is_last_token": cls_is_last_token,
		},
        "feedforward": {
            "input_dim": transformer_size,
            "num_layers": 1,
            "hidden_dims": transformer_size,
            "activations": "tanh"
        },
        "dropout": dropout,
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": batch_size
        }
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
    },
}
