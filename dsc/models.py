from typing import Any
import alignnet


import hydra
import itertools
import os
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torchaudio
import yaml
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl

from omegaconf import DictConfig
from torch import nn
from alignnet import mean_pooling

from .optimizer import OptimizerWrapper

try:
    import fairseq
except ImportError:
    warnings.warn("Fairseq not installed - wav2vec unavailable")


# ------------------------
# Audio Processing Modules
# ------------------------
class Wav2Vec(nn.Module):
    def __init__(
        self, pretrain_path, training_type="fine-tune", time_pooling=mean_pooling
    ):
        """
        Pytorch wrapper for wav2vec.

        Parameters
        ----------
        pretrain_path : str
            Path to pretrained model of wav2vec
        training_type : str, optional
            How wav2vec will be trained. "full-train" allows all wav2vec weights
            to update. "fine-tune" locks down the convolutional feature extractor
            (~4 million parameters for wav2vec_base.pt). By default "fine-tune".
        time_pooling : function, optional
            How time frames are pooled. If None frame-based results will be
            returned. Defaults to mean_pooling.
        Raises
        ------
        ValueError
            _description_
        """
        expected_training_types = ["fine-tune", "full-train"]
        if training_type not in expected_training_types:
            raise ValueError(f"training_type must be one of f{expected_training_types}")
        super().__init__()
        self.time_pooling = time_pooling
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [pretrain_path]
        )
        self.wav2vec = model[0]
        # Likely only helps by reducing size of object - sets quantizer and
        # project_q modules to None (requires features_only=True when calling forward)
        self.wav2vec.remove_pretraining_modules()

        # Disable gradient updates for convolutional feature extractor if we
        # are fine-tuning
        if training_type == "fine-tune":
            self.wav2vec.feature_grad_mult = 0
            # Should keep it out of the trainable parameters count and save some
            # GPU memory
            self.wav2vec.feature_extractor.requires_grad_(False)
            # TODO: Unsure if you can set
            # self.wav2vec.feature_extractor.requires_grad_(False) but that may
            # help with memory limitations

    def forward(self, audio):
        """
        Forward method for wav2vec

        Parameters
        ----------
        audio : _type_
            _description_

        Returns
        -------
        torch.Tensor
            Feature representation of audio. If Wav2Vec.time_pooling is None,
            will contain features on frame dimension as well (e.g. feature x
            frames tensor for each audio file).
        """
        audio = audio.squeeze(dim=1)
        audio = audio.squeeze(dim=1)
        audio_representation = self.wav2vec(audio, mask=False, features_only=True)
        feature_scores = audio_representation["x"]
        if self.time_pooling is not None:
            feature_scores = self.time_pooling(feature_scores)
        return feature_scores


class LinearSequence(nn.Module):
    def __init__(
        self,
        in_features,
        n_layers=2,
        activation=nn.ReLU,
        layer_dims=None,
        last_activate=False,
    ):
        """
        Generate sequence of n_layers Fully Connected (nn.linear) layers with activation

        Parameters
        ----------
        n_layers : int
            Number of linear layers to include.
        in_features : int
            Number of features in input
        activation : nn.Module
            Activation to include between layers. There will always be
            (n_layers - 1) activations in the sequence.
        layer_dims : list
            List of layer dimensions, not including input features (these are
            specified by in_features).
        """
        # TODO - the inputs for this are poorly defined/redundant. A refactor
        # could really improve this/make more generalizable.
        super().__init__()
        if layer_dims is not None and n_layers != len(layer_dims):
            n_layers = len(layer_dims)
        self.n_layers = n_layers
        self.in_features = in_features
        self.activation = activation
        self.layer_dims = layer_dims
        self.last_activate = last_activate
        self.setup_layers(self.n_layers)

    def setup_layers(self, n_layers):
        """
        Set up and store layers into `output_layers` attribute.

        If `self.layer_dims` is not None, linear layers are made that match the
        dimension of that list. If it is None, layers are made such that the dimension
        decreases by 1/2 for each layer.

        Parameters
        ----------
        n_layers : int
            Number of layers to make if `self.layer_dims` is not defined.
        """
        n_features = self.in_features
        layers = []

        if self.layer_dims is None:
            for k in range(n_layers - 1):
                next_feat = int(n_features / 2)
                layer = nn.Linear(n_features, next_feat)
                layers.append(layer)
                layers.append(self.activation())
                n_features = next_feat
            # Final layer to map to MOS
            layers.append(nn.Linear(n_features, 1))
        else:
            for k, layer_dim in enumerate(self.layer_dims):
                # Map previous number of features to layer dim
                layer = nn.Linear(n_features, layer_dim)
                layers.append(layer)
                if k < len(self.layer_dims) - 1:
                    layers.append(self.activation())
                elif k == len(self.layer_dims) - 1 and self.last_activate:
                    layers.append(self.activation())

                # Save current layer dim as previous number of features
                n_features = layer_dim

        self.output_layers = nn.ModuleList(layers)

    def forward(self, frame_scores):
        """
        Forward method for fully connected linear sequence.

        Parameters
        ----------
        frame_scores : torch.tensor
            Input tensor for linear sequence.

        Returns
        -------
        torch.Tensor
            Frame-based representation of audio (e.g., features x frames tensor
            for each audio file).
        """
        for k, layer in enumerate(self.output_layers):
            frame_scores = layer(frame_scores)
        return frame_scores


class Wav2VecLinearSequence(nn.Module):
    def __init__(
        self,
        pretrain_path,
        n_layers,
        training_type="fine-tune",
        activation=nn.ReLU,
        time_pooling=mean_pooling,
    ):
        """
        Wav2Vec layer into a sequence of fully connected (lienar) layers.

        Parameters
        ----------
        pretrain_path : _type_
            _description_
        n_layers : _type_
            _description_
        training_type : str, optional
            _description_, by default "fine-tune"
        activation : _type_, optional
            _description_, by default nn.ReLU
        time_pooling : _type_, optional
            _description_, by default mean_pooling
        """
        super().__init__()
        self.wav2vec = Wav2Vec(
            pretrain_path=pretrain_path,
            training_type=training_type,
            time_pooling=time_pooling,
        )

        in_features = self.wav2vec.wav2vec.post_extract_proj.out_features
        max_layers = np.floor(np.log10(in_features) / np.log10(2))
        if n_layers > max_layers:
            warnings.warn(
                (
                    f"More layers passed than useful, {n_layers} passed, max is "
                    f"{max_layers}, fixing to max."
                )
            )
            n_layers = max_layers
        self.linear_sequence = LinearSequence(
            n_layers=n_layers, in_features=in_features, activation=activation
        )

    def forward(self, audio):
        """

        Parameters
        ----------
        audio : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        frame_scores = self.wav2vec(audio)
        frame_scores = self.linear_sequence(frame_scores)
        # Feel like the way this is working right now is not great
        if len(frame_scores.shape) == 1:
            frame_scores = frame_scores[None, :]

        return frame_scores


# -------------
# MOSNet blocks
# -------------
"""
Below has MOSNet code and implementation (pulled from alignnet)
"""


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.3):
        """
        Convolutional block used in MOSNet.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        dropout : float, optional
            Dropout probability, by default 0.3
        """
        super().__init__()
        self.block = nn.Sequential(
            # Input shape: (B, T, in_channels, N)
            # Output shape: (B, T, out_channels, ceil(N/3)) (stride 3 in freq
            # of last convolutional layer causes decrease)
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 3),
                padding=(1, 1),
            ),
            nn.Dropout(0.3),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class MOSNet(nn.Module):
    def __init__(
        self,
    ):
        """
        Implement MOSNet architecture, mostly as described in "MOSNet: Deep
        Learning based Objective Assessment for Voice Conversion" by Lo et al. (2019).

        Unlike the original, this implementation does not implement frame-level loss.
        """
        super().__init__()
        self.convolutions = nn.Sequential(
            # Input shape: (B, 1, T, 257)
            # Output shape: (B, 16, T, 86) (stride 3 in freq of last convolutional
            # layer causes decrease)
            ConvBlock(1, 16),
            # Input shape: (B, 16, T, 86)
            # Output shape: (B, 32, T, 29)
            ConvBlock(16, 32),
            # Input shape: (B, 32, T, 29)
            # Output shape: (B, 64, T, 10)
            ConvBlock(32, 64),
            # Input shape: (B, 64, T, 10)
            # Output shape: (B, 128, T, 4)
            ConvBlock(64, 128),
        )
        self.blstm = nn.LSTM(
            # input_size - number of expected features in input x
            input_size=512,  # 4*128
            # hidden_size - number of features in hidden state h
            hidden_size=128,
            # num_layers - number of recurrent layers
            num_layers=1,
            # bias - bool if bias weight used (defaults to True)
            # batch_first - if True, then input and output tensors are provided
            # as (batch, seq, feature) instead of (seq, batch, feature)
            batch_first=True,
            # dropout
            # bidirectional
            bidirectional=True,
        )
        # (B, T, 256=2*128), 2 b/c bidirectional==True, 128 b/c hidden_size=128
        # and proj_size=0
        self.fc = nn.Sequential(
            nn.Linear(
                in_features=256,
                out_features=128,
            ),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(
                in_features=128,
                out_features=1,
            ),
        )
        self.time_pooling = torch.mean

    def forward(self, x):
        # x dim: (B, C, T, F) = (B, T, 1, 257)
        y = self.convolutions(x)
        # y dim: (B, C, T, F) = (B, 128, T, 4)
        # Swap dimensions to preserve frame-level time before flattening for BLSTM
        y = torch.movedim(y, -2, -3).flatten(start_dim=-2, end_dim=-1)
        # y dim: (B, T, F*C): (B, T, 512)
        y, _ = self.blstm(y)
        # y dim: (B, T, 2*H): (B, T, 256) -- H is hidden dimension, 2x b/c Bidirectional
        y = self.fc(y)
        # y dim: (B, T, 1)
        y = self.time_pooling(y, dim=1)
        # y dim: (B, 1)
        # y = torch.squeeze(y)
        # # y dim: B
        return y


# -----------------------
# Aligner - Dataset Alignment Modules
# -----------------------
class NoAligner(nn.Module):
    def __init__(self, reference_index=0, num_datasets=0, **kwargs):
        """
        NoAligner acts as a dummy module so that the other AlignNet module code
        can be used even when there is no dataset alignment being performed.

        Parameters
        ----------
        reference_index : int, optional
            Unused, but exists to easily replace other Aligner setups, by default 0
        num_datasets : int, optional
            Unused, but exists to easily replace other Aligner setups, by default 0
        """
        super().__init__()
        self.reference_index = reference_index
        self.num_datasets = num_datasets

    def forward(self, stilde, dataset_index):
        return stilde


class LinearSequenceAligner(nn.Module):
    def __init__(
        self,
        reference_index,
        num_datasets,
        embedding_dim=10,
        layer_dims=[16, 16, 16, 16, 1],
    ):
        """
        Aligner network for dataset alignment.

        The LinearSequenceAligner implements the Aligner as defined in "AlignNet:
        Learning dataset score alignment functions to enable better training of
        speech quality estimators."

        Parameters
        ----------
        reference_index : int
            Dataset index that should be treated as the reference. The Aligner acts
            as the identity function on the reference dataset.
        num_datasets : int
            Number of datasets in training.
        embedding_dim : int, optional
            Size of the dataset index embedding, by default 10
        layer_dims : list, optional
            Dimensions of the Aligner's fully connected layers, by default
            [16, 16, 16, 16, 1]
        """
        super().__init__()
        self.reference_index = reference_index
        self.num_datasets = num_datasets

        self.embedding_dim = embedding_dim
        self.embedding = torch.nn.Embedding(
            num_datasets, embedding_dim=self.embedding_dim
        )
        self.weights = LinearSequence(
            in_features=self.embedding_dim + 1,
            layer_dims=layer_dims,
        )

    def forward(self, stilde, dataset_index):
        embed = self.embedding(dataset_index)
        # Concatenate score with embedding
        x = torch.cat((stilde, embed), 1)
        score = self.weights(x)
        # Override for reference dataset -- this might be weird!
        out = torch.where(dataset_index[:, None] == self.reference_index, stilde, score)
        return out


# ------------------
# Combination Module
# ------------------
class AlignNet(nn.Module):
    def __init__(
        self, audio_net, aligner, aligner_corr_threshold=None, audio_net_freeze_epochs=0
    ):
        """
        AlignNet module that uses an audio_net with an aligner to train on
        multiple datasets at once.

        Parameters
        ----------
        audio_net : nn.Module
            Network component that maps audio to quality on the reference dataset scale.
        aligner : nn.Module
            Network component that maps intermediate quality estimates and dataset
            indicators to the appropriate dataset score.
        aligner_corr_threshold : float, optional
            Correlation threshold that determines when the aligner is activated.
            If None, the aligner turns on immediately, by default None
        audio_net_freeze_epochs : int, optional
            Number of epochs to keep the audio_net frozen, by default 0
        """
        super().__init__()
        self.audio_net = audio_net
        self.aligner = aligner
        self.reference_index = self.aligner.reference_index

        if aligner_corr_threshold is not None and aligner_corr_threshold > -1:
            # We want to freeze aligner (and ideally ensure it is not changing
            # estimations) until we see a validation correlation above
            # aligner_corr_threshold.
            self.use_aligner = False
            self.aligner_corr_threshold = aligner_corr_threshold

            # Freeze aligner params
            for p in self.aligner.parameters():
                p.requires_grad_(False)

        else:
            self.use_aligner = True
            self.aligner_corr_threshold = -1

        self.audio_net_freeze_epochs = audio_net_freeze_epochs

        if audio_net_freeze_epochs > 0:
            self.set_audio_net_update_status(False)
        else:
            self.update_audio_net = True

    def set_audio_net_update_status(self, status):
        self.update_audio_net = status
        for p in self.audio_net.parameters():
            p.requires_grad_(status)

    def forward(self, audio, dataset):
        # Intermediate score representation
        score = self.audio_net(audio)
        if self.use_aligner:
            # Aligned score estimate
            score = self.aligner(score, dataset)
        return score


# --------------
# Primary Module
# --------------
class Model(pl.LightningModule):
    def __init__(
        self,
        network: nn.Module,
        loss=torch.nn.MSELoss(),
        optimizer=torch.optim.Adam,
        loss_weights=None,
    ):
        """
        LightningModule to train AlignNet models.

        Module should be compatible with non-AlignNet architecture but includes
        additional functionality specifically tailored to AlignNet.

        Parameters
        ----------
        network : nn.Module
            AlignNet model.
        loss : func, optional
            Loss function, by default torch.nn.MSELoss()
        optimizer : OptimizerWrapper or torch.nn.optim class, optional
            Optimizer class, by default torch.optim.Adam
        loss_weights : list
            List of weights to compute weighted average of loss over datasets.
            If None, then loss is computed without respect to datasets. In the
            case where one dataset has significantly less data, a weighted
            average allows more control to ensure it is properly learned. If
            loss_weights = 1, then all the datasets will receive equal weight.
        """
        super().__init__()
        self.network = network

        self.loss = loss
        if loss_weights == 1:
            n_datasets = self.network.aligner.num_datasets
            loss_weights = n_datasets * [1 / n_datasets]
        self.loss_weights = loss_weights
        self.reset_training_step_info()
        self.optimizer = optimizer
        self.validation_step_info = {
            "outputs": [],
            "targets": [],
            "datasets": [],
        }
        self.epoch = 0

    def loss_calc(self, mean_estimate, mos, dataset):
        """
        Perform loss calculation, taking into account loss weights.

        Parameters
        ----------
        mean_estimate : torch.tensor
            Network estimate.
        mos : torch.tensor
            Labeled truth value.
        dataset : torch.tensor
            Dataset indicators.

        Returns
        -------
        torch.tensor
            Loss.
        """
        # If there are loss weights, use them
        if self.loss_weights is not None:
            loss = 0
            for dix in torch.unique(dataset):
                dix = int(dix)
                weight = self.loss_weights[dix]
                sub_ests = mean_estimate[dataset == dix]
                sub_mos = mos[dataset == dix]
                loss += weight * self.loss(sub_ests, sub_mos)

        else:
            loss = self.loss(mean_estimate, mos)
        return loss

    def forward(self, audio, dataset):
        mean_estimate = self.network(audio, dataset)
        mean_estimate = torch.squeeze(mean_estimate, dim=1)
        return mean_estimate

    def _forward(self, training_batch, is_training=False):
        """
        Internal forward method with logic consistent across all training and test steps

        Parameters
        ----------
        training_batch : tuple
            All data in a training batch.
        """
        audio, mos, dataset = training_batch
        mos = mos.float()

        est_ref_scale = self.network.audio_net(audio)
        if self.network.use_aligner:
            # Aligner is active so we use it
            mean_estimate = self.network.aligner(est_ref_scale, dataset)
        else:
            # Aligner not on yet so ignore it
            mean_estimate = est_ref_scale

        # Do we need this?
        est_ref_scale = torch.squeeze(est_ref_scale, dim=1)
        mean_estimate = torch.squeeze(mean_estimate, dim=1)

        if is_training:
            # Store min and max observed for each dataset (used for plotting alignments)
            for ds in torch.unique(dataset):
                dix = dataset == ds
                d_est = est_ref_scale[dix]
                # Update min and max for dataset as necessary
                if self.training_step_info[ds]["min"] > d_est.min():
                    self.training_step_info[ds]["min"] = d_est.min().detach()
                if self.training_step_info[ds]["max"] < d_est.max():
                    self.training_step_info[ds]["max"] = d_est.max().detach()

        loss = self.loss_calc(mean_estimate, mos, dataset)

        if mos.shape == torch.Size([1]):
            warnings.warn(f"Batch only has one element, reporting correlation=0")
            corrcoef = 0
        else:
            corrcoef = self.pearsons_corr(mean_estimate, mos)
        return loss, corrcoef

    def pearsons_corr(self, mean_estimate, mos):
        """
        Simple wrapper for grabbing pearsons correlation coefficient
        """
        mean_estimate = torch.unsqueeze(mean_estimate, dim=1)
        mos = torch.unsqueeze(mos, dim=1)

        cat = torch.cat([mean_estimate, mos], dim=1)
        cat = torch.transpose(cat, 0, 1)

        # Really lazy solution
        if len(cat.shape) > 2:
            cat = cat[:, :, 0]
        corrcoef = torch.corrcoef(cat)[0, 1]

        return corrcoef

    def training_step(self, training_batch, idx):
        loss, corrcoef = self._forward(training_batch, is_training=True)
        self.log("train_loss", loss)
        self.log("train_pearsons", corrcoef)
        return loss

    def validation_step(self, val_batch, idx):
        """
        Validation step. Unlike the training and test steps, we need to store
        per-dataset information here.
        """

        audio, mos, dataset = val_batch
        mos = mos.float()

        mean_estimate = self.network(audio, dataset)
        mean_estimate = torch.squeeze(mean_estimate, dim=1)

        loss = self.loss_calc(mean_estimate, mos, dataset)

        # Store per dataset information to be used at epoch end.
        self.validation_step_info["outputs"].append(mean_estimate)
        self.validation_step_info["targets"].append(mos)
        self.validation_step_info["datasets"].append(dataset)

        return loss

    def on_validation_epoch_end(self) -> None:
        """
        At the end of validation epochs we calculate per-dataset statistics.
        """
        # TODO - add logic for saving alignments
        # Concatenate stored epoch data into single tensor for each metric
        estimates = torch.cat(self.validation_step_info["outputs"], dim=0)
        targets = torch.cat(self.validation_step_info["targets"], dim=0)
        datasets = torch.cat(self.validation_step_info["datasets"], dim=0)

        # Overall loss and correlation
        loss = self.loss_calc(estimates, targets, datasets)
        corrcoef = self.pearsons_corr(estimates, targets)

        # Check if network has a use_aligner flag
        if hasattr(self.network, "use_aligner"):
            # If aligner is off and we have passed the correlation threshold, do
            #  the updates
            if (
                not self.network.use_aligner
                and corrcoef > self.network.aligner_corr_threshold
            ):
                # Start using alignment network in forward
                self.network.use_aligner = True
                # Turn on gradients for aligner parameters
                for p in self.network.aligner.parameters():
                    p.requires_grad_(True)
                print(
                    (
                        f"Correlation threshold {self.network.aligner_corr_threshold} "
                        f"reached with {corrcoef:.4f}. Turning on aligner."
                    )
                )
        if self.epoch > 0:
            # Plot the learned alignments
            self.plot_alignments()
            # Reset the range of values seen on the ref/target scales that epoch
            self.reset_training_step_info()

        self.log("val_loss", loss)
        self.log("val_pearsons", corrcoef)

        # Per dataset losses and correlations
        for k, ds in enumerate(torch.unique(datasets)):
            ds_ix = datasets == ds

            ds_est = estimates[ds_ix]
            ds_tar = targets[ds_ix]

            ds_loss = self.loss(ds_est, ds_tar)
            ds_corr = self.pearsons_corr(ds_est, ds_tar)
            self.log(f"val_loss/dataset {k}", ds_loss)
            self.log(f"val_pearsons/dataset {k}", ds_corr)

        # Clear epoch validation info
        for k, v in self.validation_step_info.items():
            v.clear()

        # If we aren't updating audio-net and our epoch has passed the wait time,
        # turn it on!
        if (
            not self.network.update_audio_net
            and self.epoch >= self.network.audio_net_freeze_epochs
        ):
            # Turn on audio_net, set
            print(f"Turning audio_net on after {self.epoch} epochs.")
            self.network.set_audio_net_update_status(True)
        self.epoch += 1

        return super().on_validation_epoch_end()

    def reset_training_step_info(self):
        num_datasets = self.network.aligner.num_datasets
        # Initialize array to store min and max observed on the reference scale
        # for each dataset.
        self.training_step_info = [
            {"min": torch.inf, "max": -torch.inf} for _ in range(num_datasets)
        ]

    def plot_alignments(self):
        for dataset, data_range in enumerate(self.training_step_info):
            data_min = data_range["min"]
            data_max = data_range["max"]
            n_vals = 500
            step = (data_max - data_min) / n_vals
            ref_scale_probe = torch.arange(data_min, data_max, step)
            datasets = torch.tensor(len(ref_scale_probe) * [dataset])

            device = data_min.device
            ref_scale_probe = ref_scale_probe.to(device)
            datasets = datasets.to(device)
            # get alignments
            target_scale_probe = self.network.aligner(
                ref_scale_probe[:, None], datasets
            )
            # Move to CPU
            target_scale_probe = target_scale_probe.detach().to("cpu")
            ref_scale_probe = ref_scale_probe.detach().to("cpu")

            # Plot alignmetns
            plt.plot(ref_scale_probe, target_scale_probe)
        plt.xlabel("reference scale")
        plt.ylabel("target scale")
        plt.title(f"Alignments - epoch: {self.epoch:03d}")
        plt.show()

    def test_step(self, test_batch, idx):
        loss, corrcoef = self._forward(test_batch)
        self.log("test_loss", loss)
        self.log("test_pearsons", corrcoef)
        return loss

    def configure_optimizers(self):
        if isinstance(self.optimizer, OptimizerWrapper):
            optimizer = self.optimizer.optimizer(self.parameters())
        else:
            optimizer = self.optimizer(self.parameters())
        return optimizer
