# Dataset-Concealment
This repository is in support of the ICASSP 2026 submission titled "Unseen but not Unknown: Using Dataset Concealment to Robustly Evaluate Speech Quality Estimation Models."

# Installation
```
conda env create -f environment.yml
conda activate dsc
pip install .
```

In order to run and create an environment with package versions exactly matching those used to run tests for the paper run
```
conda env create -f environment-paper.yml
conda activate dsc-paper
pip install .
```

# Data Preparation
In order to quickly define the Individual, Concealed, and Global dataset groups and train models with them, use the `prep-dataset-groups.py` CLI.

## Example
The file `example_dataset_input.yaml` has an example of what a dataset group dictionary looks like.
It is formatted as
```
"Dataset1": "path/to/dataset1"
"Dataset2": "path/to/dataset2"
"Dataset3": "path/to/dataset3"
```

It is expected that each dataset path is structured as follows:
```bash
path/to/dataset1
├── test.csv
├── train.csv
└── valid.csv
```

Running
```
python prep-dataset-groups.py example_dataset_input.yaml --output-dir data-groups
```
will generate all the required dataset group files in the `dsc/config/data/data_dirs` folder.
The `data_dirs` folder has the following structure.
```bash
dsc/config/data/data_dirs/
├── Concealed
│   ├── Conceal-Dataset1.yaml
│   ├── Conceal-Dataset2.yaml
│   └── Conceal-Dataset3.yaml
├── Global.yaml
├── Individual
│   ├── Dataset1.yaml
│   ├── Dataset2.yaml
│   └── Dataset3.yaml
```

# Training Details
The following enables Global training on the first optimization device availble. 
This may need to be updated depending on your configuration.

Wav2Vec
```
python train.py \
data/data_dirs=Global \
'optimization.devices=[0]' \
--config-dir dsc/config/models/ \
--config-name alignnet-wav2vec
```

NISQA
```
python train.py \
data/data_dirs=Global \
'optimization.devices=[0]' \
--config-dir dsc/config/models/ \
--config-name alignnet-nisqa
```

MOSNet
```
python train.py \
data/data_dirs=Global \
'optimization.devices=[0]' \
--config-dir dsc/config/models/ \
--config-name alignnet-MOSNet
```

## Training Concealed or Individual models in a multirun
The following example demonstrates quickly training all the concealed dataset groups.
The logic for other models or individual dataset groups is similar.
```
python train.py -m \
data/data_dirs=Concealed/Conceal-Dataset1,Concealed/Conceal-Dataset2,Concealed/Conceal-Dataset3 \
'optimization.devices=[0]' \
--config-dir dsc/config/models/ \
--config-name alignnet-nisqa
```

# Information Regarding Paper Results
The config files in the `dsc/config` directory detail all the configuration details used to train models to generate results for the paper "Unseen but not Unknown: Using Dataset Concealment to Robustly Evaluate Speech Quality Estimation Models."

## Example models
The `trained_model` folder contains three trained, example model checkpoints used in the paper. 
The results from the paper come from 10 independently trained versions of each model, so the included checkpoints cannot fully replicate results on their own.
They are included for convenience.