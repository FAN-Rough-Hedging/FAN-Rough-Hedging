DATASET_SETING = {
    'MODE': 'stimulation',
    'OPTION_TYPE': 'euro',
    'PATH': None,
    'BATCH_SIZE': 1024,
}

OPTIMIZER_SETTING = {
    'NAME': 'AdamW',
    'ARGS' : {
        'LR': 1e-3,
        'BETAS': (0.9, 0.999),
    },
    'LR_SCHEDULER': {
        'NAME': 'StepLR',
        'STEP_SIZE': 10,
        'GAMMA': 0.98,
    },
    'EPOCHS': 1000,
}

FAN_SETTING = {
    'DATASET': DATASET_SETING,
    'OPTIMIZER' : OPTIMIZER_SETTING,
    'MODEL_HYPERPARAMETERS': {
        'series_len':128,
		'hurst':0.1,
		'd_v':256,
		'd_emb':256,
    }
}

LSTM_SETTING = {
    'DATASET': DATASET_SETING,
    'OPTIMIZER' : OPTIMIZER_SETTING,
    'MODEL_HYPERPARAMETERS': {
        'input_dim':1,
    }
}

VANILLAT_TRANSFORMER_SETTING = {
    'DATASET': DATASET_SETING,
    'OPTIMIZER' : OPTIMIZER_SETTING,
    'MODEL_HYPERPARAMETERS': {
        'input_dim':1,
    }
}