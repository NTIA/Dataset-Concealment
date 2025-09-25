# Dataset-Concealment
This repository is in support of the ICASSP 2026 submission titled "Unseen but not Unknown: Using Dataset Concealment to Robustly Evaluate Speech Quality Estimation Models."

# Installation
Main envrironment
```
conda env create -f environment.yml
conda activate dsc
pip install .
```


TODO - update this with more detailed install instructions (should actually all be good with the environment files now I think --- they each use pip install from relevant githubs directly.)
* AlignNets code: https://github.com/NTIA/alignnet
* NISQA code: https://github.com/gabrielmittag/NISQA
    * Only NISQA_lib.py is required.
* Wav2Vec2.0 FAIRSEQ fork: https://github.com/JPieper-NTIA/fairseq
    * Minor updates to make it easier to train Wav2Vec2.0 for speech quality estimation in more modern python environments.

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
will generate all the required dataset group files in the `data-groups` folder.
The `data-groups` folder has the following structure.
```bash
data-groups/
├── Concealed
│   ├── Conceal-Dataset1.yaml
│   ├── Conceal-Dataset2.yaml
│   └── Conceal-Dataset3.yaml
├── Global.yaml
└── Individual
    ├── Dataset1.yaml
    ├── Dataset2.yaml
    └── Dataset3.yaml
```

# Training Details
Wav2Vec
```
python train.py \
data/data_dirs=quick-debug.yaml \
'optimization.devices=[0]' \
--config-dir dsc/config/models/ \
--config-name alignnet-wav2vec
```

NISQA
```
python train.py \
data/data_dirs=quick-debug.yaml \
'optimization.devices=[0]' \
--config-dir dsc/config/models/ \
--config-name alignnet-nisqa
```

MOSNet
```
python train.py \
data/data_dirs=quick-debug.yaml \
'optimization.devices=[0]' \
--config-dir dsc/config/models/ \
--config-name alignnet-MOSNet
```

```
python ~/projects/nr-ml-speech-quality-toolbox/train.py -m \
project.name='Dataset Alignment/debug' \
project.task=mosnet \
data/data_dirs=holdouts/hold-NISQA \
data.split=0 \
network.aligner.reference_index=0 \
network.aligner_corr_threshold=-1 \
'optimization.devices=[0]' \
--config-dir ~/projects/nr-ml-speech-quality-toolbox/nr_toolbox/config-alignnet/models/ \
--config-name alignnet-MOSNet.yaml
```