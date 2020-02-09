local bert_model = "bert-base-uncased";
local learning_rate = 5e-6;

{
    "train_data_path": "aita/aita-train.pkl",
    "validation_data_path": "aita/aita-dev.pkl",

    "dataset_reader": {
        "type": "aita_bert_simple_reader",
        "lazy": false,
        "only_title": true,
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": bert_model,
            "do_lowercase": true
        },
        "token_indexers": {
            "bert": {
                "type": "bert-pretrained",
                "pretrained_model": bert_model,
            }
        }
    },

    "iterator": {
        "type": "bucket",
        "batch_size": 6,
        "sorting_keys": [["tokens", "num_tokens"]],
    },

    "trainer": {
        "num_epochs": 120,
        "cuda_device": 0,
        "validation_metric": "+accuracy",
        "optimizer": {
            "type": "bert_adam",
            "lr": learning_rate,
        },
        "num_serialized_models_to_keep": 1,
        "grad_norm": 1.0,
    },

    "model": {
        "type": "bert_for_classification",
        "bert_model": bert_model,
        "dropout": 0.1,
        "num_labels": 4,
    },
}