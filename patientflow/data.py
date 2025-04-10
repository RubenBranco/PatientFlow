from enum import Enum
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Iterator, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import LightningDataModule
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, random_split


class FeatureTemporality(Enum):
    STATIC = "static"
    TEMPORAL = "temporal"


class Feature:
    def __init__(
        self,
        feature_name: str,
        temporality: Literal[FeatureTemporality.STATIC, FeatureTemporality.TEMPORAL],
    ):
        """
        Represents information regarding a feature in a dataset.

        Args:
            `feature_name` (str): The name of the feature.

            `temporality` (Literal[FeatureTemporality.STATIC,
            FeatureTemporality.TEMPORAL]): The temporality of the feature. If the
                feature is static, use `FeatureTemporality.STATIC`. If the feature is
                temporal, use `FeatureTemporality.TEMPORAL`.
        """
        self.name = feature_name
        self.temporality = temporality

    def __repr__(self):
        return self.__str__()


class ContinuousFeature(Feature):
    def __init__(
        self,
        feature_name: str,
        temporality: Literal[FeatureTemporality.STATIC, FeatureTemporality.TEMPORAL],
        scaler: Optional[Any] = None,
    ):
        """
        Represents information regarding a continuous feature in a dataset.

        Args:
            `feature_name` (str): The name of the feature.

            `temporality` (Literal[FeatureTemporality.STATIC,
            FeatureTemporality.TEMPORAL]): The temporality of the feature. If the
                feature is static, use `FeatureTemporality.STATIC`. If the feature is
                temporal, use `FeatureTemporality.TEMPORAL`.

            `scaler` (Optional[Any]): An optional scaler (from scikit-learn) to use for
                scaling the feature values. If `None`, no scaling is applied.
        """
        super().__init__(feature_name, temporality)
        self.scaler = scaler

    def __str__(self):
        return f"ContinuousFeature(name={self.name}, temporality={self.temporality})"


class CategoricalFeature(Feature):
    def __init__(
        self,
        feature_name: str,
        temporal: Literal[FeatureTemporality.STATIC, FeatureTemporality.TEMPORAL],
        values: List[Any],
    ):
        """
        Represents information regarding a categorical feature in a dataset.

        Args:
            `feature_name` (str): The name of the feature.

            `temporal` (Literal[FeatureTemporality.STATIC,
            FeatureTemporality.TEMPORAL]): The temporality of the feature. If the
                feature is static, use `FeatureTemporality.STATIC`. If the feature is
                temporal, use `FeatureTemporality.TEMPORAL`.

            `values` (List[Any]): A list of possible values for the feature.

        """
        super().__init__(feature_name, temporal)
        self.values = values

    def __str__(self):
        return f"CategoricalFeature(name={self.name}, temporality={self.temporality}, values={self.values})"

    def __len__(self):
        return len(self.values)


class FeatureList:
    def __init__(self, features: Optional[List[Feature]] = None):
        self.features = features or list()

    def __getitem__(self, item: Any) -> Feature:
        return self.features[item]

    def __len__(self) -> int:
        return len(self.features)

    def __iter__(self) -> Iterator:
        return iter(self.features)

    def __contains__(self, item) -> bool:
        return item in self.features

    def __str__(self) -> str:
        return str(self.features)

    def __repr__(self) -> str:
        return str(self.features)

    def __index__(self, item) -> int:
        return self.features.index(item)

    def insert(self, index: int, feature: Feature) -> None:
        self.features.insert(index, feature)

    def index_by_name(self, name: str) -> int:
        """
        Returns the index of the feature with the given name.

        Args:
            `name` (str): The name of the feature to search for.

        Returns:
            The index of the feature with the given name.

        Raises:
            `ValueError`: If no feature with the given name is found.
        """
        for i, feature in enumerate(self.features):
            if feature.name == name:
                return i
        raise ValueError(f"Feature {name} not found.")

    def get_feature_by_name(self, name: str) -> "Feature":
        """
        Returns the feature with the given name.

        Args:
            `name` (str): The name of the feature to retrieve.

        Returns:
            The feature with the given name.

        Raises:
            `ValueError`: If no feature with the given name is found.
        """
        return self.features[self.index_by_name(name)]

    def append(self, feature) -> None:
        self.features.append(feature)

    def extend(self, features) -> None:
        self.features.extend(features)

    def feature_names(self) -> List[str]:
        """
        Returns a list of the names of all features in the dataset.

        Returns:
            A list of the names of all features in the dataset.
        """
        return [feature.name for feature in self.features]

    def static_features(self) -> "FeatureList":
        """
        Returns a new FeatureList containing only the static features in the dataset.

        Returns:
            A new FeatureList containing only the static features in the dataset.
        """
        return FeatureList(
            list(
                filter(
                    lambda feature: feature.temporality == FeatureTemporality.STATIC,
                    self.features,
                )
            )
        )

    def temporal_features(self) -> "FeatureList":
        """
        Returns a new FeatureList containing only the temporal features in the dataset.

        Returns:
            A new FeatureList containing only the temporal features in the dataset.
        """
        return FeatureList(
            list(
                filter(
                    lambda feature: feature.temporality == FeatureTemporality.TEMPORAL,
                    self.features,
                )
            )
        )

    def continuous_features(self) -> "FeatureList":
        """
        Returns a new FeatureList containing only the continuous features in the dataset.

        Returns:
            A new FeatureList containing only the continuous features in the dataset.
        """
        return FeatureList(
            list(
                filter(
                    lambda feature: isinstance(feature, ContinuousFeature),
                    self.features,
                )
            )
        )

    def categorical_features(self) -> "FeatureList":
        """
        Returns a new FeatureList containing only the categorical features in the dataset.

        Returns:
            A new FeatureList containing only the categorical features in the dataset.
        """
        return FeatureList(
            list(
                filter(
                    lambda feature: isinstance(feature, CategoricalFeature),
                    self.features,
                )
            )
        )

    def categorical_features_indices(self) -> List[int]:
        """
        Returns a list of the indices of all categorical features in the feature list.

        Returns:
            A list of the indices of all categorical features in the feature list.
        """
        return [
            i
            for i, feature in enumerate(self.features)
            if isinstance(feature, CategoricalFeature)
        ]

    def categorical_features_num_categories(self) -> List[int]:
        """
        Returns a list of the number of categories for each categorical feature in the feature list.

        Returns:
            A list of the number of categories for each categorical feature in the feature list.
        """
        return [
            len(feature.values)
            for feature in self.features
            if isinstance(feature, CategoricalFeature)
        ]


_DEFAULT_BRAINTEASER_CATEGORICAL_FEATURES = [
    # binary
    "NIV",
    "Tracheostomy",
    "PEG",
    "Weightloss_>10%",
    "ALS_familiar_history",
    "Ever_smoked",
    "Blood_hypertension",
    "Diabetes",
    "Dyslipidemia",
    "Thyroid",
    "Autoimmune",
    "Stroke",
    "Cardiac_disease",
    "Primary_cancer",
    "C9orf72",
    # multi
    "Gender",
    "Ethnicity",
    "UMNvsLMN",
    "Onset",
    "Limb_O",
    "Limbs_Impairment",
    "Limbs_Side",
    "SOD1 Mutation",
    "TARDBP mutation",
    "FUS mutation",
    "P1",
    "P2",
    "P3",
    "P4",
    "P5",
    "P6",
    "P7",
    "P8",
    "P9",
    "P10",
    "P11",
    "P12",
]

_DEFAULT_BRAINTEASER_TEMPORAL_FEATURES = [
    "P1",
    "P2",
    "P3",
    "P4",
    "P5",
    "P6",
    "P7",
    "P8",
    "P9",
    "P10",
    "P11",
    "P12",
    "medianDate",
]


class BrainteaserDataModule(LightningDataModule):
    def __init__(
        self,
        seed: int,
        batch_size: int,
        data_path: Optional[Path] = None,
        df_data: Optional[pd.DataFrame] = None,
        val_size: Optional[float] = 0.0,
        shuffle: Optional[bool] = True,
        categorical_features: Optional[List[str]] = None,
        temporal_features: Optional[List[str]] = None,
    ):
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.seed = seed
        self.batch_size = batch_size
        self.data_path = data_path
        self.df_data = df_data

        self.val_size = val_size
        self.shuffle = shuffle
        self.seq_lengths_weights = None

        self._categorical_features = (
            categorical_features or _DEFAULT_BRAINTEASER_CATEGORICAL_FEATURES
        )
        self._temporal_features = (
            temporal_features or _DEFAULT_BRAINTEASER_TEMPORAL_FEATURES
        )

        self.features = FeatureList()

    @property
    def train_static_data(self) -> Union[torch.Tensor, None]:
        if self.train_dataset is not None:
            return self.train_dataset[:][0]
        return None

    @property
    def train_temporal_data(self) -> Union[torch.Tensor, None]:
        if self.train_dataset is not None:
            return self.train_dataset[:][1]
        return None

    @property
    def train_max_steps(self) -> Union[int, None]:
        if self.train_dataset is not None:
            return self.train_dataset[0][1].size(0)
        return None

    def encode_features(
        self,
        df: pd.DataFrame,
        reassign_seq_lengths_weights: bool = False,
        requires_median_delta_calc: bool = True,
    ) -> List[pd.DataFrame]:
        if not self.features:
            for feature in df.columns.drop(
                labels=["REF", "prog_profile"], errors="ignore"
            ):
                feature_temporality = (
                    FeatureTemporality.TEMPORAL
                    if feature in self._temporal_features
                    else FeatureTemporality.STATIC
                )
                if feature in self._categorical_features:
                    feature_cat = df[feature].astype("category")
                    df[feature] = feature_cat.cat.codes
                    self.features.append(
                        CategoricalFeature(
                            feature,
                            feature_temporality,
                            feature_cat.cat.categories.to_list(),
                        )
                    )
                else:
                    self.features.append(
                        ContinuousFeature(feature, feature_temporality)
                    )

        else:
            for feature in self.features:
                if isinstance(feature, CategoricalFeature):
                    df[feature.name] = pd.Categorical(
                        df[feature.name], categories=feature.values
                    ).codes

        has_median_date = "medianDate" in df.columns
        if has_median_date and requires_median_delta_calc:
            df["medianDate"] = pd.to_datetime(df["medianDate"], format="%Y-%m-%d")

        dfs = []

        for feature in self.features:
            if feature.name == "medianDate":
                continue

            if isinstance(feature, ContinuousFeature):
                if feature.scaler is None:
                    scaler = MinMaxScaler().fit(df[feature.name].values.reshape(-1, 1))
                    feature.scaler = scaler
                else:
                    scaler = feature.scaler

                df[feature.name] = scaler.transform(
                    df[feature.name].values.reshape(-1, 1)
                )

        seq_lengths = []

        for _, patient_df in df.groupby(by="REF"):
            patient_df = patient_df.reset_index(drop=True)

            if has_median_date and requires_median_delta_calc:
                # Store the original dates to avoid losing information during calculation
                original_dates = patient_df["medianDate"].copy()

                # Calculate days between consecutive visits
                for i in range(1, len(patient_df)):
                    patient_df.loc[i, "medianDate"] = (
                        original_dates[i] - original_dates[i - 1]
                    ).days
                patient_df.loc[0, "medianDate"] = 0

            if (patient_df[[f"P{j}" for j in range(1, 13)]] == 0.0).all().all():
                continue

            seq_lengths.append(len(patient_df))
            patient_df = patient_df.astype(np.float32)
            dfs.append(patient_df)

        seq_lengths = torch.LongTensor(seq_lengths)

        if self.seq_lengths_weights is None or reassign_seq_lengths_weights:
            self.seq_lengths_weights = torch.ones(
                seq_lengths.max() + 1, dtype=torch.float32
            )

            for idx, count in zip(*seq_lengths.unique(return_counts=True)):
                self.seq_lengths_weights[idx] += count

            self.seq_lengths_weights[0] = 0.0
            self.seq_lengths_weights = (
                self.seq_lengths_weights / self.seq_lengths_weights.sum()
            )

        if has_median_date:
            median_date_feature = self.features[
                self.features.index_by_name("medianDate")
            ]

            if median_date_feature.scaler is None:
                scaler = QuantileTransformer(random_state=self.seed).fit(
                    pd.concat(dfs, ignore_index=True)["medianDate"].values.reshape(
                        -1, 1
                    )
                )
                median_date_feature.scaler = scaler
            else:
                scaler = median_date_feature.scaler

            for patient_df in dfs:
                patient_df["medianDate"] = scaler.transform(
                    patient_df["medianDate"].values.reshape(-1, 1)
                )

        return dfs

    def patient_dfs_to_tensors(
        self, dfs: List[pd.DataFrame]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        static_features = self.features.static_features().feature_names()

        static_features_tensor = torch.zeros(
            (len(dfs), len(static_features)), dtype=torch.float32
        )
        max_visits = max([len(df) for df in dfs])
        temporal_features_tensor = torch.zeros(
            (len(dfs), max_visits, len(self._temporal_features)),
            dtype=torch.float32,
        )
        seq_lengths = torch.zeros(len(dfs), dtype=torch.long)

        for i, patient_df in enumerate(dfs):
            static_features_tensor[i] = torch.FloatTensor(
                patient_df.loc[0, static_features].values
            )
            temporal_features_tensor[i, 0 : len(patient_df)] = torch.FloatTensor(
                patient_df[self._temporal_features].values
            )
            seq_lengths[i] = len(patient_df)

        return static_features_tensor, temporal_features_tensor, seq_lengths

    def setup(self, stage=None) -> None:
        if (stage == "fit" or stage is None) and self.train_dataset is None:
            # train dataset

            if self.df_data is None:
                df = pd.read_csv(self.data_path / "train.csv")
            else:
                df = self.df_data

            dfs = self.encode_features(df)

            (
                static_tensor,
                temporal_tensor,
                seq_lengths_tensor,
            ) = self.patient_dfs_to_tensors(dfs)

            patient_dataset = TensorDataset(
                static_tensor, temporal_tensor, seq_lengths_tensor
            )

            if self.val_size:
                self.train_dataset, self.val_dataset = random_split(
                    patient_dataset,
                    [
                        int(np.ceil(len(patient_dataset) * (1.0 - self.val_size))),
                        int(len(patient_dataset) * self.val_size),
                    ],
                    generator=torch.Generator().manual_seed(self.seed),
                )
            else:
                self.train_dataset = patient_dataset

            # test dataset
            if self.df_data is None:
                df = pd.read_csv(self.data_path / "test.csv")
                dfs = self.encode_features(df)
                (
                    static_tensor,
                    temporal_tensor,
                    seq_lengths_tensor,
                ) = self.patient_dfs_to_tensors(dfs)

                patient_dataset = TensorDataset(
                    static_tensor, temporal_tensor, seq_lengths_tensor
                )
                self.test_dataset = patient_dataset

    def inverse_transform_sample(
        self, x_t: Tensor, x_s: Optional[Tensor] = None
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if x_s is not None:
            for i, feature in enumerate(self.features.static_features()):
                if isinstance(feature, ContinuousFeature):
                    x_s[:, i] = torch.from_numpy(
                        feature.scaler.inverse_transform(
                            x_s[:, i].unsqueeze(1)
                        ).flatten()
                    ).flatten()

        for i, feature in enumerate(self.features.temporal_features()):
            if isinstance(feature, ContinuousFeature):
                x_t[:, :, i] = torch.from_numpy(
                    feature.scaler.inverse_transform(
                        x_t[:, :, i].flatten().unsqueeze(1)
                    )
                ).view(x_t.size(0), x_t.size(1))

        return x_s, x_t

    def sample_to_df(
        self,
        x_s: Tensor,
        x_t: Tensor,
        seq_lengths: Tensor,
        transform_categorical: bool = True,
        transform_continuous: bool = True,
    ) -> pd.DataFrame:
        dfs = []

        if transform_continuous:
            x_s, x_t = self.inverse_transform_sample(torch.clone(x_t), torch.clone(x_s))

        for i in range(x_s.size(0)):
            for timestep in range(seq_lengths[i]):
                patient = {"REF": [i]}
                for j, feature in enumerate(self.features.static_features()):
                    if (
                        isinstance(feature, CategoricalFeature)
                        and transform_categorical
                    ):
                        patient[feature.name] = [
                            feature.values[x_s[i, j].long().item()]
                        ]
                    else:
                        patient[feature.name] = [x_s[i, j].item()]

                for j, feature in enumerate(self.features.temporal_features()):
                    if (
                        isinstance(feature, CategoricalFeature)
                        and transform_categorical
                    ):
                        patient[feature.name] = [
                            feature.values[x_t[i, timestep, j].long().item()]
                        ]
                    else:
                        patient[feature.name] = [x_t[i, timestep, j].item()]
                dfs.append(pd.DataFrame(patient))

        return pd.concat(dfs, ignore_index=True)

    def df_to_endpoint_tensor_dataset(
        self, df: pd.DataFrame, k: int, endpoint: int, scale_time_delta: bool = True
    ):
        static_tensors = []
        temporal_tensors = []
        labels = []
        seq_lengths = []

        max_visits = max([len(pdf) for _, pdf in df.groupby("REF")])

        for _, patient_df in list(df.groupby("REF")):
            patient_df = patient_df.reset_index(drop=True)
            if endpoint == 1:
                # C1 need for NIV P12 <= 3 which means should be between 1 and 4
                # assuming df is not in original scale
                if (patient_df["P12"] == -1).all():
                    continue
                p_endpoint = patient_df["P12"].between(1, 4)
            elif endpoint == 2:
                # C2 need for auxiliary communication device P1 <= 1 which means should be between 1 and 2
                # assuming df is not in original scale
                if (patient_df["P1"] == -1).all():
                    continue
                p_endpoint = patient_df["P1"].between(1, 2)
            elif endpoint == 3:
                # C3 need for percutaneous endoscopic gastronomy (PEG) P3 <= 2 which means should be between 1 and 3
                # assuming df is not in original scale
                if (patient_df["P3"] == -1).all():
                    continue
                p_endpoint = patient_df["P3"].between(1, 3)
            elif endpoint == 4:
                # C4 need for a caregiver P5 <= 1 or P6 <= 1 which means should be between 1 and 2
                # assuming df is not in original scale
                if (patient_df["P5"] == -1).all() and (patient_df["P6"] == -1).all():
                    continue
                p_endpoint = patient_df["P5"].between(1, 2) | patient_df["P6"].between(
                    1, 2
                )
            elif endpoint == 5:
                # C5 need for a wheelchair P8 <= 1 which means should be between 1 and 2
                # assuming df is not in original scale
                if (patient_df["P8"] == -1).all():
                    continue
                p_endpoint = patient_df["P8"].between(1, 2)

            endpoint_window = p_endpoint.idxmax() if p_endpoint.any() else None

            if scale_time_delta:
                patient_df["medianDate"] = self.features.temporal_features()[
                    -1
                ].scaler.inverse_transform(
                    patient_df["medianDate"].values.reshape(-1, 1)
                )
            endpoint_time = (
                sum(
                    [
                        patient_df.loc[i]["medianDate"]
                        for i in range(0, endpoint_window + 1)
                    ]
                )
                if p_endpoint.any()
                else None
            )

            for i, window_df in enumerate(patient_df.expanding(1)):
                if endpoint_window is not None and i >= endpoint_window:
                    break

                static_features_tensor = torch.FloatTensor(
                    window_df.loc[
                        0, self.features.static_features().feature_names()
                    ].values
                )
                label = (
                    int(
                        sum(
                            [
                                window_df.iloc[j]["medianDate"]
                                for j in range(0, len(window_df))
                            ]
                        )
                        + k
                        >= endpoint_time
                    )
                    if endpoint_time is not None
                    else 0
                )
                temporal_features_tensor = torch.zeros(
                    (max_visits, len(self.features.temporal_features())),
                    dtype=torch.float32,
                )

                temporal_features_tensor[0 : len(window_df)] = torch.FloatTensor(
                    window_df[self._temporal_features].values
                )

                if scale_time_delta:
                    temporal_features_tensor[:, -1] = torch.from_numpy(
                        self.features.temporal_features()[-1].scaler.inverse_transform(
                            temporal_features_tensor[:, -1].flatten().unsqueeze(1)
                        )
                    ).flatten()

                static_tensors.append(static_features_tensor)
                temporal_tensors.append(temporal_features_tensor)
                labels.append(label)
                seq_lengths.append(len(window_df))

        if not static_tensors:
            return None

        static_tensor = torch.stack(static_tensors)
        temporal_tensor = torch.stack(temporal_tensors)
        labels_tensor = torch.FloatTensor(labels)
        seq_lengths_tensor = torch.LongTensor(seq_lengths)

        return TensorDataset(
            static_tensor, temporal_tensor, seq_lengths_tensor, labels_tensor
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=cpu_count(),
            persistent_workers=True,
            multiprocessing_context="spawn",
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=cpu_count(),
            persistent_workers=True,
            multiprocessing_context="spawn",
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=cpu_count(),
            persistent_workers=True,
            multiprocessing_context="spawn",
        )
