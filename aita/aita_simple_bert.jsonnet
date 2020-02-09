local train_path = "/home/johnkeszler/Documents/leddit/aita/aita-train.pkl";
local val_path = "/home/johnkeszler/Documents/leddit/aita/aita-dev.pkl";


local bert_model = "bert-base-uncased";
local batch_size = std.extVar("BATCH_SIZE");
local learning_rate = std.extVar("LEARNING_RATE");
local dropout = std.extVar("DROPOUT");

{
    "train_data_path": train_path,
    "validation_data_path": val_path,

    "dataset_reader": {
        "type": "aita_bert_simple_reader",
        "lazy": false,
        "only_title": false,
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
        "type": "basic",
        "batch_size": batch_size,
    },

    "trainer": {
        "num_epochs": 5,
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
        "dropout": dropout,
        "num_labels": 4,
    },
}
