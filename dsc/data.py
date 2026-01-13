import os
import pickle
import re
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
import torchaudio
import warnings

import numpy as np
import pandas as pd
import pytorch_lightning as pl

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from . import NISQA_lib as NL
import alignnet.transforms as transforms


class AudioData(Dataset):
    def __init__(
        self,
        data_files,
        transform=None,
        transform_time="get",
        cache=False,
        target="MOS",
        target_transform=None,
        pathcol="full_path",
        data_path="",
        fs=16000,
        percent=1,
    ):
        """
        Class for loading audio data with quality (or any target) scores.

        Parameters
        ----------
        data_files : list
            List of paths to data files containing audio paths and target scores.
        transform : _type_, optional
            Transform object for applying transforms to audio, by default None
        transform_time : str, optional
            When to perform transforms, by default "get"
        cache : bool, optional
            Store audio rather than loading it each time, by default False
        target : str, optional
            Target to train to, must be column name in data files, by default "MOS"
        pathcol : str, optional
            Name of column containing audio path in data files, by default "full_path"
        data_path : str, optional
            Parent path to data for paths stored in pathcol, by default "".
        fs : int, optional
            Sampling rate required for the network. All audio will be resampled
            to this sample rate if necessary. If fs is set to None, the resampling
            step will be skipped.
        """
        self.fs = fs
        self.percent = percent

        score_files = list()
        for k, file in enumerate(data_files):
            score_file = pd.read_csv(file)
            if self.percent < 1:
                n_items = len(score_file)
                keep_items = np.ceil(n_items * self.percent).astype(int)
                rng = np.random.default_rng()
                keep_ix = rng.choice(len(score_file), keep_items, replace=False)
                score_file = score_file.loc[keep_ix]
            score_file["Dataset_Indicator"] = k
            score_files.append(score_file)

        self.score_file = pd.concat(score_files, ignore_index=True)
        if cache:
            self.wavs = dict()

        self.data_path = data_path
        self.pathcol = pathcol
        self.target = target
        self.target_transform = target_transform
        self.transform = transform
        self.transform_time = transform_time

    def __len__(self):
        return len(self.score_file)

    def __getitem__(self, idx):
        dataset = self.score_file.loc[idx, "Dataset_Indicator"]
        if self.target is not None:
            mos = self.score_file.loc[idx, self.target]
        else:
            mos = torch.nan
        if self.target_transform is not None:
            mos = self.target_transform(mos, dataset)

        audio_path = os.path.join(
            self.data_path, self.score_file.loc[idx, self.pathcol]
        )
        if hasattr(self, "wavs") and audio_path in self.wavs:
            # Already loaded and cached the audio
            audio = self.wavs[audio_path]
        else:
            # Load audio
            audio, sample_rate = torchaudio.load(audio_path)

            # Resample as needed
            if self.fs is not None and sample_rate != self.fs:
                resampler = torchaudio.transforms.Resample(
                    sample_rate, self.fs, dtype=audio.dtype
                )
                audio = resampler(audio)
                sample_rate = self.fs

            if self.transform is not None and self.transform_time == "get":
                # Apply transform

                if self.fs is None:
                    # We are not resampling which means the transform should handle this
                    audio = self.transform.transform(audio, fs=sample_rate)
                else:
                    # We have resampled, so the transform doesn't need to account for it
                    audio = self.transform.transform(audio)

            # TODO figure out if this is the right place to do this
            audio = audio.float()
            if hasattr(self, "wavs"):
                self.wavs[audio_path] = audio
        return audio, mos, dataset

    def padding(self, batch):
        audio_files, mos, dataset = zip(*batch)
        mos = torch.tensor(np.array(mos))
        dataset = torch.tensor(dataset)

        time_dim = 0

        max_len = np.max([audio.shape[time_dim] for audio in audio_files])
        if max_len == 1:
            time_dim = 1
            max_len = np.max([audio.shape[time_dim] for audio in audio_files])
        # # TODO don't think we need this?
        # audio_shape = audio_files[0].size()

        audio_out = []

        for ix, audio in enumerate(audio_files):
            if audio.shape[time_dim] < max_len:
                repeat_samples = max_len - audio.shape[time_dim]
                # Initialize pad width for each dimension to no padding
                pad_width = [(0, 0) for i in range(len(audio.shape))]
                # Dimension always time - pad end with repeat_samples
                pad_width[time_dim] = (0, repeat_samples)
                # Convert to tuple for input to np.pad
                pad_width = tuple(pad_width)

                audio = np.pad(audio, pad_width=pad_width, mode="constant")
                audio = torch.from_numpy(audio)
            audio_out.append(audio)

        # Concatenate into one tensor
        audio_out = torch.stack(audio_out, dim=0)
        if self.transform is not None and self.transform_time == "collate":
            audio_out = self.transform.transform(audio_out)
        audio_out = torch.unsqueeze(audio_out, dim=1)
        return audio_out, mos, dataset


class FeatureData(AudioData):
    """
    For loading pre-computed features for audio files. Only the __getitem__
    method needs to change
    """

    def __init__(
        self,
        flatten=True,
        dim_cutoff=None,
        dim=0,
        **kwargs,
    ):
        """
        Class for pre-computed features of audio files.

        Parameters
        ----------
        flatten : bool, optional
            Flatten representation into a single dimension, by default True
        dim_cutoff : _type_, optional
            Max number of dimensions to consider. Useful if want to restrict
            number of mel bands in an FMS representation where there are
            multiple sample rates present. By default None.
        dim : int, optional
            Dimension on which to perform cutoff using dim_cutoff, by default 0
        """
        super().__init__(
            **kwargs,
        )
        self.dim_cutoff = dim_cutoff
        self.dim = dim
        self.flatten = flatten

    def __getitem__(self, idx):
        dataset = self.score_file.loc[idx, "Dataset_Indicator"]
        mos = self.score_file.loc[idx, self.target]
        if self.target_transform is not None:
            mos = self.target_transform(mos, int(dataset))

        audio_path = os.path.join(
            self.data_path, self.score_file.loc[idx, self.pathcol]
        )
        if hasattr(self, "wavs") and audio_path in self.wavs:
            # Already loaded and cached the audio
            audio = self.wavs[audio_path]
        else:
            fname, ext = os.path.splitext(audio_path)
            # If using same split csvs as audio this may say wav and not pt
            # (coming out of pretransform_data.py will save as pt)
            if ext == ".wav":
                audio_path = fname + ".pkl"
            # Load audio
            with open(audio_path, "rb") as feat_input:
                audio = pickle.load(feat_input)
            # audio = torch.load(audio_path)
            if self.dim_cutoff is not None:
                audio = torch.narrow(
                    audio, dim=self.dim, start=0, length=self.dim_cutoff
                )

            if self.flatten:
                # Flatten by column
                audio = audio.t().flatten()

            if self.transform is not None and self.transform_time == "get":
                # Apply transform
                audio = self.transform.transform(audio)

            # TODO figure out if this is the right place to do this
            audio = audio.float()
            if hasattr(self, "wavs"):
                self.wavs[audio_path] = audio
        # # TODO Delete this
        # audio = audio - (-4.0185)
        return audio, mos, dataset


class CSVFeatureData(Dataset):
    """
    For loading pre-computed features for audio files that are all stored in one csv.
    Only the __getitem__ method needs to change.
    """

    def __init__(
        self,
        data_files,
        feature_cols,
        target_col=-1,
        percent=1,
        header=None,
        transform=None,
    ):
        """
        Class for pre-computed features of audio files stored in a single csv file.

        Parameters
        ----------
        data_files : list
            List of paths to data files containing audio paths and target scores.
        TODO feature_cols:
        transform : _type_, optional
            Transform object for applying transforms to audio, by default None
        transform_time : str, optional
            When to perform transforms, by default "get"
        TODO target : str, optional
            Target to train to, must be column name in data files, by default "MOS"
        """
        self.percent = percent

        score_files = list()
        for k, file in enumerate(data_files):
            if header is None:
                score_file = pd.read_csv(file, header=header)
            else:
                score_file = pd.read_csv(file)
            if self.percent < 1:
                n_items = len(score_file)
                keep_items = np.ceil(n_items * self.percent).astype(int)
                rng = np.random.default_rng()
                keep_ix = rng.choice(len(score_file), keep_items, replace=False)
                score_file = score_file.loc[keep_ix]
            score_file["Dataset_Indicator"] = k
            score_files.append(score_file)

        self.score_file = pd.concat(score_files, ignore_index=True)

        self.target_col = target_col
        if not isinstance(feature_cols, list) and not isinstance(
            feature_cols, np.ndarray
        ):
            feature_cols = eval(feature_cols)
        self.feature_cols = feature_cols

    def __len__(self):
        return len(self.score_file)

    def __getitem__(self, idx):
        # TODO
        dataset = self.score_file.loc[idx, "Dataset_Indicator"]
        mos = self.score_file.loc[idx, self.target_col]

        audio = self.score_file.loc[idx, self.feature_cols].to_numpy()

        audio = torch.Tensor(audio)
        return audio, mos, dataset


class WawenetsAudioData(AudioData):
    def __init__(
        self,
        target="PESQMOSLQO",
        target_transform=None,
        pathcol="filename",
        **kwargs,
    ):
        """
        Simple wrapper for AudioData class that constructs full paths for wawenets
        audio and stores it in the score_file.
        """

        super().__init__(
            target=target,
            target_transform=target_transform,
            pathcol=pathcol,
            **kwargs,
        )

        self.make_fullpath()

    def make_fullpath(self):
        fullpaths = []
        for ix, row in self.score_file.iterrows():
            fname = row[self.pathcol]
            fname = os.path.basename(fname)
            fileid, _ = os.path.splitext(fname)

            file_parts = fileid.split("_")
            # Last part of the filename is DYXX where Y is series and XX is source
            subdir = file_parts[-1]
            # Remove D
            subdir = subdir[1:]

            file_path = os.path.join(subdir, fname)
            fullpaths.append(file_path)
        self.pathcol = "full_path"
        self.score_file[self.pathcol] = fullpaths


class WawenetsFeatureData(WawenetsAudioData, FeatureData):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)


class MOSNetAudioData(AudioData):
    def __init__(
        self,
        fft_win_length=512,
        fft_win_overlap=256,
        target_fs=16000,
        fft_include_phase=False,
        flatten=False,
        smart_save=False,
        *args,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fft_win_length = fft_win_length
        self.fft_win_overlap = fft_win_overlap
        self.target_fs = target_fs
        self.fft_include_phase = fft_include_phase
        self.flatten = flatten
        self.smart_save = smart_save
        # Smart save must be a string that will be the parent path where we save
        # all the features we compute within. The full data path will follow
        # whatever smart_save is.
        transform_params = [p for p in dir(self) if "fft" in p or "target_fs" in p]
        if self.smart_save:
            self.data_id = ""
            for p in transform_params:
                val = getattr(self, p)
                self.data_id += f"{p}_{val}-"
            self.data_id = self.data_id[:-1]
            self.feature_path = os.path.join(smart_save, self.data_id)
            os.makedirs(self.feature_path, exist_ok=True)
        # Initialize transform
        self.transform = transforms.STFTTransform()
        # Assign relevant values to transform
        for p in transform_params:
            if hasattr(self.transform, p):
                val = getattr(self, p)
                setattr(self.transform, p, val)

    def __getitem__(self, idx):
        dataset = self.score_file.loc[idx, "Dataset_Indicator"]
        if self.target is not None:
            mos = self.score_file.loc[idx, self.target]
        else:
            mos = torch.nan

        if self.target_transform is not None:
            mos = self.target_transform(mos, int(dataset))

        audio_path = os.path.join(
            self.data_path, self.score_file.loc[idx, self.pathcol]
        )
        if hasattr(self, "feature_path"):
            # Check if we already computed this representation
            if audio_path[0] == "/":
                # Need to remove leading slash if we start with abspath
                feature_path = os.path.join(self.feature_path, audio_path[1:])
            else:
                feature_path = os.path.join(self.feature_path, audio_path)
        else:
            feature_path = None

        if hasattr(self, "wavs") and audio_path in self.wavs:
            # Already loaded and cached the audio
            audio = self.wavs[audio_path]
        elif os.path.exists(feature_path):
            # If feature path exists, load it
            with open(feature_path, "rb") as f:
                audio = pickle.load(f)
        else:
            # Load audio
            audio, sample_rate = torchaudio.load(audio_path)

            # Resample as needed
            if self.target_fs is not None and sample_rate != self.target_fs:
                resampler = torchaudio.transforms.Resample(
                    sample_rate, self.target_fs, dtype=audio.dtype
                )
                audio = resampler(audio)
                sample_rate = self.target_fs

            if self.transform is not None and self.transform_time == "get":
                # Apply transform
                audio = self.transform.transform(audio)

            # TODO figure out if this is the right place to do this
            audio = audio.float()
            if hasattr(self, "wavs"):
                self.wavs[audio_path] = audio
            # Store feature representaion if we are doing that
            if feature_path is not None:
                ppath = os.path.dirname(feature_path)
                os.makedirs(ppath, exist_ok=True)
                with open(feature_path, "wb") as f:
                    pickle.dump(audio, f)
        # This needs to happen outside of the loading and transforming audio
        if self.flatten:
            # Flatten by column
            audio = audio.t().flatten()

        return audio, mos, dataset


class NISQAAudioData(AudioData):
    def __init__(
        self,
        ms_seg_length=15,
        max_length=None,
        ms_seg_hop_length=1,
        ms_n_fft=1024,
        ms_hop_length=80,
        ms_win_length=170,
        ms_n_mels=32,
        ms_sr=48e3,
        ms_fmax=16e3,
        ms_channel=None,
        ms_max_segments=1300,
        n_outs=1,
        smart_save=False,
        *args,
        **kwargs,
    ):
        """
        NISQA requires its own librosa style loading for the audio files.
        It also requires information about `n_wins` for its forward methods.
        This is because it has real time processing, so they mask out the padded
        audio within batches and only focus on the real signal when doing
        self attention.

        In the __get_item__ function this needs to smush the audio and n_wins into
        a single tuple. Then I adapted the NISQA_DIM.forward method to grab element
        0 as x and element 1 as n_wins.
        """
        super().__init__(**kwargs)
        self.ms_seg_length = ms_seg_length
        self.max_length = max_length
        self.ms_seg_hop_length = ms_seg_hop_length
        self.ms_n_fft = ms_n_fft
        self.ms_hop_length = ms_hop_length
        self.ms_win_length = ms_win_length
        self.ms_n_mels = ms_n_mels
        self.ms_sr = ms_sr
        self.ms_fmax = ms_fmax
        self.ms_channel = ms_channel
        self.ms_max_segments = ms_max_segments
        self.n_outs = n_outs
        self.smart_save = smart_save
        # Smart save must be a string that will be the parent path where we save
        # all the features we compute within. The full data path will follow
        # whatever smart_save is.
        if self.smart_save:
            self.data_id = ""
            transform_params = [p for p in dir(self) if "ms" in p and "seg" not in p]
            for p in transform_params:
                val = getattr(self, p)
                self.data_id += f"{p}_{val}-"
            self.data_id = self.data_id[:-1]
            self.feature_path = os.path.join(smart_save, self.data_id)
            os.makedirs(self.feature_path, exist_ok=True)

    def __getitem__(self, idx):
        dataset = self.score_file.loc[idx, "Dataset_Indicator"]

        if self.target is not None:
            mos = self.score_file.loc[idx, self.target]
        else:
            mos = torch.nan
        if self.n_outs > 1:
            mos = self.n_outs * [mos]

        if self.target_transform is not None:
            mos = self.target_transform(mos, dataset)

        audio_path = os.path.join(
            self.data_path, self.score_file.loc[idx, self.pathcol]
        )
        if hasattr(self, "feature_path"):
            # Check if we already computed this representation
            if audio_path[0] == "/":
                # Need to remove leading slash if we start with abspath
                feature_path = os.path.join(self.feature_path, audio_path[1:])
            else:
                feature_path = os.path.join(self.feature_path, audio_path)
        else:
            feature_path = None

        if hasattr(self, "wavs") and audio_path in self.wavs:
            # Already loaded and cached the audio
            spec = self.wavs[audio_path]

        elif os.path.exists(feature_path):
            # If it exists load it
            with open(feature_path, "rb") as f:
                spec = pickle.load(f)
        else:
            # Load audio
            spec = NL.get_librosa_melspec(
                audio_path,
                sr=self.ms_sr,
                n_fft=self.ms_n_fft,
                hop_length=self.ms_hop_length,
                win_length=self.ms_win_length,
                n_mels=self.ms_n_mels,
                fmax=self.ms_fmax,
                ms_channel=self.ms_channel,
            )

            # Cache if we caching
            if hasattr(self, "wavs"):
                self.wavs[audio_path] = spec
            # Store feature representation if we are doing that
            if feature_path is not None:
                ppath = os.path.dirname(feature_path)
                os.makedirs(ppath, exist_ok=True)
                with open(feature_path, "wb") as f:
                    pickle.dump(spec, f)
        # TODO -- they have a conditional on self.seg_length is not None but I
        # don't think we need that right now
        x_spec_seg, n_wins = NL.segment_specs(
            audio_path,
            spec,
            self.ms_seg_length,
            self.ms_seg_hop_length,
            self.ms_max_segments,
        )

        audio = (x_spec_seg, n_wins)
        return audio, mos, dataset

    def padding(self, batch):
        # Split out audio/mos/dataset
        audio_files, mos, dataset = zip(*batch)
        # Split out audio from n_wins
        audio_files, n_wins = zip(*audio_files)
        n_wins = torch.tensor(np.array(n_wins))

        # Do audio padding with normal padding logic
        simple_batch = [item for item in zip(audio_files, mos, dataset)]
        audio_files, mos, dataset = super().padding(simple_batch)
        # Unsure where this extra dimension is introduced but handling it here
        audio_files = audio_files.squeeze(dim=1)

        # Reconstruct audio to have n-wins
        audio_files = [item for item in zip(audio_files, n_wins)]

        return audio_files, mos, dataset


class NISQAFeatureData(NISQAAudioData, FeatureData):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # TODO -- make sure you're not missing anything here


# TODO: Lots of repeated defaults...make globals or something so that only have
# to change defaults in one spot...
class AudioDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dirs,
        batch_size=16,
        num_workers=1,
        persistent_workers=True,
        DataClass=AudioData,
        collate_type="padding",
        data_percent=1,
        split=None,
        inference_dirs=None,
        **kwargs,
    ):
        """
        Primary audio data module that prepares data for training, testing, or
        predictions.

        Parameters
        ----------
        data_dirs : list
            List of paths to directories containing train.csv, valid.csv, and test.csv
            for each dataset.
        batch_size : int, optional
            Number of items in each batch, by default 32
        num_workers : int, optional
            Number of workers used during training, by default 1
        persistent_workers : bool, optional
            Whether or not workers persist between epochs, by default True
        DataClass : class, optional
            Class that the data will be initialized with. Assumed to inherit from
            torch.utils.data.Dataset, by default AudioData
        collate_type : str, optional
            String that determines what type of collate function is used, by default
            "padding"
        data_percent : float, optional
            Float between 0 and 1 to use only a fraction of the given data.
        split : int, optional
            If multiple splits of a given dataset exist in a directory and are named
            splitXX will attempt to find the appropriate split regardless of what split
            is specified by the paths in data_dirs.
        inference_dirs : list, optional
            List of paths to csv files containing data to evaluate with a trained model.
        **kwargs : optional
            Additional arguments are passed to the DataClass when instantiated in
            AudioDataModule.setup()

        """
        super().__init__()

        # If this class sees batch_size=auto, it sets to default value and
        # assumes a Tuner is being called in the main logic to update this later
        if batch_size == "auto":
            batch_size = 32
        self.batch_size = batch_size
        self.collate_type = collate_type
        self.DataClass = DataClass
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers

        self.data_class_kwargs = kwargs

        data_dirs = self.update_data_dir_splits(data_dirs, split)
        self.data_dirs = data_dirs
        self.inference_dirs = inference_dirs

    def update_data_dir_splits(self, data_dirs, split):
        """
        Update splits if they are given

        Parameters
        ----------
        data_dirs : _type_
            _description_
        split : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        # Handle dataset splits if they are given
        if split is not None:
            # We have a split --- regex to find it in data paths
            split_search = re.compile(r"(split\d{2})")
            # Split we want to replace with
            split = f"split{split:02d}"
            for ix, data_dir in enumerate(data_dirs):
                search = split_search.search(data_dir)
                if search:
                    # If we have the split find the split string
                    orig_split = search.groups()[0]
                    # Replace
                    data_dir = data_dir.replace(orig_split, split)
                    data_dirs[ix] = data_dir
            print(f"data_dirs updated:\n{data_dirs}")
        return data_dirs

    def setup(self, stage: str):
        """
        Load different datasubsets depending on stage.

        If stage == 'fit', then train, valid, and test data are loaded.

        If stage == 'test', then only test data is loaded.

        If stage == 'predict', then self.data_dirs should be full paths to the specific
        csv files to run predictions on.

        Parameters
        ----------
        stage : str
            One of fit, test, or predict.
        """
        if stage == "fit":
            train_paths = self.find_datasubsets(self.data_dirs, "train")
            self.train = self.DataClass(
                data_files=train_paths,
                **self.data_class_kwargs,
            )

            valid_paths = self.find_datasubsets(self.data_dirs, "valid")
            self.valid = self.DataClass(
                data_files=valid_paths,
                **self.data_class_kwargs,
            )

            test_paths = self.find_datasubsets(self.data_dirs, "test")
            self.test = self.DataClass(
                data_files=test_paths,
                **self.data_class_kwargs,
            )
        elif stage == "test":
            test_paths = self.find_datasubsets(self.data_dirs, "test")
            self.test = self.DataClass(
                data_files=test_paths,
                **self.data_class_kwargs,
            )
        elif stage == "predict":
            if self.inference_dirs is None:
                # If we only passed in data_dirs we need to assign them to
                # inference_dirs
                self.inference_dirs = self.data_dirs

            self.predict = self.DataClass(
                data_files=self.inference_dirs,
                **self.data_class_kwargs,
            )

    def find_datasubsets(self, data_paths, subset):
        """
        Find subsets as determined by data_paths and subset.

        Primarily relies on find_datasubset. Loads in train.csv, valid.csv, and
        test.csv as necessary.

        Parameters
        ----------
        data_paths : list
            List of paths for each dataset.
        subset : str
            Which type of csv file to read in.

        Returns
        -------
        list
            Paths to that datasubset.
        """
        outs = []
        for data_path in data_paths:
            out = self.find_datasubset(data_path, subset)
            outs.append(out)
        return outs

    def find_datasubset(self, data_path, subset):
        """
        Helper function for setup to find the different data subsets (train/valid/test)

        Parameters
        ----------
        data_path : str
            Path to folder containg subset.csv
        subset : str
            String for representative datasubset file in data_path

        Returns
        -------
        str
            path to subset.csv

        Raises
        ------
        ValueError
            subset could not be found in data_path.
        """
        _, ext = os.path.splitext(data_path)
        if ext == ".csv":
            return data_path
        files = os.listdir(data_path)
        out = []
        for file in files:
            basename, ext = os.path.splitext(file)
            if basename == subset:
                out.append(file)
        if len(out) == 0:
            raise ValueError(f"Unable to find {subset} in {data_path}")
        elif len(out) > 1:
            warnings.warn(
                (
                    f"Multiple matches for {subset} in {data_path}.\n"
                    "Of {out} using {out[0]}"
                )
            )
            out = out[0]
        else:
            out = out[0]
        out = os.path.join(data_path, out)
        return out

    def train_dataloader(self):
        """
        Prepare DataLoader for training.
        """
        if self.collate_type == "padding":
            collate_fn = self.train.padding
        else:
            collate_fn = None
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        """
        Prepare DataLoader for validation.
        """
        if self.collate_type == "padding":
            collate_fn = self.valid.padding
        else:
            collate_fn = None
        return DataLoader(
            self.valid,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            # for any contrastive loss shuffling validation batches will provide more
            # robust results
            shuffle=True,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        """
        Prepare DataLoader for testing.
        """
        if self.collate_type == "padding":
            collate_fn = self.test.padding
        else:
            collate_fn = None
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            collate_fn=collate_fn,
        )

    def predict_dataloader(self):
        """
        Prepare DataLoader for prediction.
        """
        if self.collate_type == "padding":
            collate_fn = self.predict.padding
        else:
            collate_fn = None
        return DataLoader(
            self.predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            collate_fn=collate_fn,
        )
