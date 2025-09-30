import hydra
import os
import torch
import warnings

import numpy as np
import pandas as pd

from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

import alignnet

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


@hydra.main(
    config_path="./inference_configs", config_name="nisqa.yaml", version_base=None
)
def main(cfg: DictConfig) -> None:
    """
    Run inference on data with a trained model.

    See `python inference.py --help` for more details.
    """

    # Transform
    transform = hydra.utils.instantiate(cfg.transform)
    data_class = hydra.utils.instantiate(cfg.dataclass)

    print("Initializing data")
    audio_data = hydra.utils.instantiate(
        cfg.data,
        transform=transform,
        DataClass=data_class,
    )

    audio_data.setup("predict")

    # load the dataloader
    dataloader = audio_data.predict_dataloader()

    print(f"Loading model from {cfg.model.path}")
    model = alignnet.load_model(cfg.model.path)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if cfg.model.dataset_index == "reference":
        dataset_index = torch.tensor([model.network.aligner.reference_index])
    else:
        dataset_index = torch.tensor([cfg.model.dataset_index])
    dataset_index = dataset_index.to(device)
    # Switch to eval mode
    model.eval()
    with torch.no_grad():
        output_dicts = []
        print(f"Generating estimations")
        for ix, (audio, mos, dataset) in enumerate(tqdm(dataloader)):
            if isinstance(audio, list):
                if len(audio) == 1:
                    audio = [(audio[0][0].to(device), audio[0][1].to(device))]

            else:
                audio = audio.to(device)

            est = model(audio, dataset_index)
            audio_path = audio_data.predict.score_file.loc[
                ix, audio_data.predict.pathcol
            ]
            output_dicts.append(
                {
                    "file": audio_path,
                    "estimate": est.to("cpu").numpy()[0],
                    "dataset": dataset.numpy()[0],
                    "AlignNet dataset index": dataset_index.to("cpu").numpy()[0],
                }
            )
    output_df = pd.DataFrame(output_dicts)
    print("First 5 results:")
    print(output_df.head())
    output_dir = os.path.dirname(cfg.output.file)

    if not os.path.exists(output_dir) and output_dir != "":
        os.makedirs(output_dir)
    print(f"Saving results to {cfg.output.file}")
    output_df.to_csv(cfg.output.file, index=False)


if __name__ == "__main__":
    main()
