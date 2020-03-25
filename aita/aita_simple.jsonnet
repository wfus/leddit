{
    "train_data_path": "aita/aita-train.pkl",
    "validation_data_path": "aita/aita-dev.pkl",

    "dataset_reader": {
        "type": "aita_simple_reader",
    },

    "iterator": {
        "type": "bucket",
        "batch_size": 64,
        "sorting_keys": [["title", "num_tokens"]],
    },

    "trainer": {
        "num_epochs": 40,
        "patience": 10,
        "cuda_device": 0,
        "grad_clipping": 5.0,
        "validation_metric": "+accuracy",
        "optimizer": {
            "type": "adagrad"
        },
    },

    "model": {
        "type": "aita_simple_model",

        "text_field_embedder": {
            "tokens": {
                "type": "embedding",
                "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.100d.txt.gz",
                "embedding_dim": 100,
                "trainable": false
            },
        },

        "text_encoder": {
            "type": 'lstm',
            'bidirectional': 'true',
            'input_size': 100,
            'hidden_size': 80,
            'num_layers': 4,
            "dropout": 0.2,
        },

        "classifier_feedforward": {
            "input_dim": 320,
            "num_layers": 3,
            "hidden_dims": [100, 32, 3],
            "activations": ['relu', 'relu', 'linear'],
            "dropout": [0.2, 0.2, 0],
        },
    },

}