DATASET_SETING = {
    'MODE': 'stimulation',
    'OPTION_TYPE': 'euro',
    'PATH': None,
    'BATCH_SIZE': 8196,
}

OPTIMIZER_SETTING = {
    'NAME': 'AdamMax',
    'ARGS' : {
        'LR': 1e-5,
        'BETAS': (0.9, 0.99),
    },
    'LR_SCHEDULER': {
        'NAME': 'CosineAnnealingWarmRestarts',
        'T_0': 500,
        'T_MULT': 2,
        'ETA_MIN': 1e-8,
    },
    'EPOCHS': 10000,
}

FAN_SETTING = {
    'DATASET': DATASET_SETING,
    'OPTIMIZER' : OPTIMIZER_SETTING,
    'MODEL_HYPERPARAMETERS': {
        'series_len':128,
		'hurst':0.1,
		'd_v':1024,
		'd_emb':1024,
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