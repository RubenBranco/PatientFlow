from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from lightning.pytorch import (
    LightningModule,
)
from torch import Tensor, nn
from torch.optim import Optimizer

from patientflow.data import CategoricalFeature, ContinuousFeature, Feature, FeatureList


class FeatureEmbedder(nn.Module):
    def __init__(
        self,
        features: FeatureList,
        embedding_dim: int,
        continuous_encoding_freq_length: int,
    ):
        super().__init__()
        self.features = features
        self.embedding_dim = embedding_dim
        self.continuous_encoding_freq_length = continuous_encoding_freq_length

        self.categorical_feature_embedders = nn.ModuleDict()
        self.continuous_feature_embedders = nn.ModuleDict()

        self.create_embedding_layers(
            features, embedding_dim, continuous_encoding_freq_length
        )

    def forward(self, x, feature: Feature) -> Tensor:
        if isinstance(feature, CategoricalFeature):
            return self.categorical_feature_embedders[feature.name](x.long())
        elif isinstance(feature, ContinuousFeature):
            freq_encoding = self.freq_encode_feature(x)
            return self.continuous_feature_embedders[feature.name](freq_encoding)
        else:
            raise ValueError(f"Unknown feature type {type(feature)}")

    def create_embedding_layers(
        self,
        features: FeatureList,
        embedding_dim: int,
        continuous_encoding_freq_length: int,
    ) -> None:
        self.create_categorical_embedding_layer(
            features.categorical_features(), embedding_dim
        )
        self.create_continuous_embedding_layer(
            features.continuous_features(),
            embedding_dim,
            continuous_encoding_freq_length,
        )

    def create_categorical_embedding_layer(
        self, features: FeatureList, embedding_dim: int
    ) -> None:
        for feature in features:
            self.categorical_feature_embedders[feature.name] = nn.Embedding(
                len(feature),
                embedding_dim,
            )

    def create_continuous_embedding_layer(
        self,
        features: FeatureList,
        embedding_dim: int,
        continuous_encoding_freq_length: int,
    ) -> None:
        for feature in features:
            self.continuous_feature_embedders[feature.name] = nn.Sequential(
                nn.Linear(2 * continuous_encoding_freq_length, embedding_dim),
                nn.SiLU(),
                nn.Linear(embedding_dim, embedding_dim),
            )

    def freq_encode_feature(self, value: Tensor) -> Tensor:
        angles = (
            2 ** torch.arange(0, self.continuous_encoding_freq_length).float()
            * torch.pi
        ).type_as(value)

        if value.dim() == 2:
            out_dim = (value.size(0), value.size(1), -1)
        else:
            out_dim = (value.size(0), -1)

        value = value.flatten().unsqueeze(-1)

        sines = torch.sin(angles * value)
        cosines = torch.cos(angles * value)

        return torch.stack((sines, cosines), dim=-1).view(*out_dim)


class AttentionPooling(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.attn_weights = nn.Linear(embedding_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        attn_scores = self.attn_weights(x)
        attn_scores = self.softmax(attn_scores)
        return (attn_scores * x).sum(dim=1)


class MeanPooling(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, x: Tensor) -> Tensor:
        return x.mean(dim=1)


class MaxPooling(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, x: Tensor) -> Tensor:
        return x.max(dim=1).values


class MLPBlock(nn.Module):
    def __init__(self, embedding_dim: int, hidden_size: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class StaticEmbeddedEncoder(nn.Module):
    def __init__(
        self,
        features: FeatureList,
        transformer_encoder_n_heads: int,
        transformer_encoder_dim_forward: int,
        embedding_dim: int,
        continuous_encoding_freq_length: int,
        pooling: Literal["mean", "max", "attn"] = "mean",
        dropout_p: float = 0.1,
        act_fn=nn.Tanh,
        variational: bool = False,
    ):
        super().__init__()

        self.features = features
        self.transformer_encoder_n_heads = transformer_encoder_n_heads
        self.transformer_encoder_dim_forward = transformer_encoder_dim_forward
        self.embedding_dim = embedding_dim
        self.continuous_encoding_freq_length = continuous_encoding_freq_length
        self.dropout_p = dropout_p
        self.act_fn = act_fn
        self.variational = variational

        self.feature_embedder = FeatureEmbedder(
            features, embedding_dim, continuous_encoding_freq_length
        )

        self.transformer_encoder = nn.TransformerEncoderLayer(
            embedding_dim,
            transformer_encoder_n_heads,
            dim_feedforward=transformer_encoder_dim_forward,
            batch_first=True,
            dropout=dropout_p,
        )

        if pooling == "attn":
            self.pooling = AttentionPooling(embedding_dim)
        elif pooling == "mean":
            self.pooling = MeanPooling(embedding_dim)
        elif pooling == "max":
            self.pooling = MaxPooling(embedding_dim)
        else:
            raise ValueError(f"Unknown pooling method {pooling}")

        self.output_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            self.act_fn(),
        )

        if variational:
            self.fc_mu = nn.Linear(embedding_dim, embedding_dim)
            self.fc_logvar = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        embs = [
            self.feature_embedder(x[:, i], feature)
            for i, feature in enumerate(self.features)
        ]
        embs = torch.stack(embs, dim=1)
        x = self.transformer_encoder(embs)
        x = self.pooling(x)

        if self.variational:
            mu = self.fc_mu(x)
            logvar = self.fc_logvar(x)
            emb = self.reparametrize(mu, logvar)
            return emb, mu, logvar

        return self.output_mlp(x)

    def reparametrize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: Tensor) -> Tensor:
        return self(x)


class TemporalEmbedderEncoder(nn.Module):
    def __init__(
        self,
        features: FeatureList,
        hidden_size: int,
        embedding_dim: int,
        num_channels: int,
        transformer_encoder_n_heads: int,
        transformer_encoder_dim_forward: int,
        num_transformer_encoder_layers: int,
        continuous_encoding_freq_length: int,
        rnn_type: Literal["lstm", "gru"] = "gru",
        dropout_p: float = 0.1,
        act_fn=nn.SiLU,
        variational: bool = False,
    ):
        super().__init__()

        self.features = features
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.num_channels = num_channels
        self.transformer_encoder_n_heads = transformer_encoder_n_heads
        self.transformer_encoder_dim_forward = transformer_encoder_dim_forward
        self.num_transformer_encoder_layers = num_transformer_encoder_layers
        self.continuous_encoding_freq_length = continuous_encoding_freq_length
        self.rnn_type = rnn_type
        self.dropout_p = dropout_p
        self.act_fn = act_fn
        self.variational = variational

        self.feature_embedder = FeatureEmbedder(
            features, embedding_dim, continuous_encoding_freq_length
        )
        self.mlp_block = MLPBlock(len(self.features) * embedding_dim, hidden_size)
        self.conv_1 = nn.Conv1d(1, self.num_channels, 1)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                embedding_dim,
                transformer_encoder_n_heads,
                dim_feedforward=transformer_encoder_dim_forward,
                dropout=dropout_p,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=num_transformer_encoder_layers,
        )
        self.conv_2 = nn.Conv1d(self.num_channels, 1, 1)

        if rnn_type == "lstm":
            self.rnn = nn.LSTM(
                hidden_size,
                embedding_dim,
                batch_first=True,
                dropout=dropout_p,
            )
        else:
            self.rnn = nn.GRU(
                hidden_size,
                embedding_dim,
                batch_first=True,
                dropout=dropout_p,
            )

        if variational:
            self.fc_mu = nn.Linear(embedding_dim, embedding_dim)
            self.fc_logvar = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        # B, T, NF, F
        embs = torch.empty(
            (x.size(0), x.size(1), len(self.features), self.embedding_dim)
        ).type_as(x)

        for i, feature in enumerate(self.features):
            embs[:, :, i] = self.feature_embedder(x[:, :, i], feature)

        # concatenate to B, T, NF * F
        embs = embs.view(x.size(0), x.size(1), -1)
        # B, T, F
        embs = self.mlp_block(embs)

        # First convolutional layer
        # B, C=1, T, F
        embs = embs.unsqueeze(1)

        # lets save original shape
        b, c, t, f = embs.size()
        # B, C=1, T*F
        embs = embs.view(b, c, -1)
        embs = self.conv_1(embs).view(b, self.num_channels, t, f)

        embs = embs.permute(0, 2, 1, 3).contiguous().view(b * t, self.num_channels, f)
        embs = (
            self.transformer_encoder(embs)
            .view(b, t, self.num_channels, f)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(b, self.num_channels, -1)
        )
        embs = self.conv_2(embs).squeeze(1).view(b, t, -1)

        embs, _ = self.rnn(embs)

        if self.variational:
            mu = self.fc_mu(embs)
            logvar = self.fc_logvar(embs)
            embs = self.reparametrize(mu, logvar)
            return embs, mu, logvar

        return embs

    def reparametrize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: Tensor) -> Tensor:
        return self(x)


class DecoderOutput:
    def __init__(self):
        self.continuous_index = []
        self.binary_index = []
        self.categorical_index = []
        self.features = []

    def __iter__(self):
        return iter(self.features)

    def __getitem__(self, key):
        return self.features[key]

    def __len__(self):
        return len(self.features)

    def add_continuous(self, values: Tensor, feature: ContinuousFeature):
        self.continuous_index.append(len(self.features))
        self.features.append((feature, values))

    def add_binary(self, values: Tensor, feature: CategoricalFeature):
        self.binary_index.append(len(self.features))
        self.features.append((feature, values))

    def add_categorical(self, values: Tensor, feature: CategoricalFeature):
        self.categorical_index.append(len(self.features))
        self.features.append((feature, values))

    def add_feature(self, values: Tensor, feature: Feature):
        if isinstance(feature, ContinuousFeature):
            self.add_continuous(values, feature)
        elif isinstance(feature, CategoricalFeature):
            if len(feature) == 2:
                self.add_binary(values, feature)
            else:
                self.add_categorical(values, feature)
        else:
            raise ValueError(f"Unknown feature type {type(feature)}")

    def get_continuous(self) -> List[Tuple[int, ContinuousFeature, Tensor]]:
        return [(i, *self.features[i]) for i in self.continuous_index]

    def get_binary(self) -> List[Tuple[int, CategoricalFeature, Tensor]]:
        return [(i, *self.features[i]) for i in self.binary_index]

    def get_categorical(self) -> List[Tuple[int, CategoricalFeature, Tensor]]:
        return [(i, *self.features[i]) for i in self.categorical_index]


class StaticEmbedderDecoder(nn.Module):
    def __init__(self, embedding_dim: int, hidden_size: int, features: FeatureList):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.features = features
        self.mlp = MLPBlock(embedding_dim, hidden_size)
        self.feature_layers = nn.ModuleDict()

        for feature in features:
            if isinstance(feature, ContinuousFeature):
                self.feature_layers[feature.name] = nn.Sequential(
                    nn.Linear(hidden_size, 1),
                    nn.Sigmoid(),
                )
            elif isinstance(feature, CategoricalFeature):
                if len(feature) == 2:
                    self.feature_layers[feature.name] = nn.Sequential(
                        nn.Linear(hidden_size, 1),
                    )
                else:
                    self.feature_layers[feature.name] = nn.Sequential(
                        nn.Linear(hidden_size, len(feature)),
                    )

    def forward(
        self, x: Tensor, transform_and_unify_output: bool = False
    ) -> Union["DecoderOutput", Tensor]:
        x = self.mlp(x)

        if transform_and_unify_output:
            outs = torch.zeros((x.size(0), len(self.features))).type_as(x)
        else:
            outs = DecoderOutput()

        for i, feature in enumerate(self.features):
            x_post = self.feature_layers[feature.name](x)

            if transform_and_unify_output:
                if isinstance(feature, CategoricalFeature):
                    if len(feature) == 2:
                        outs[:, i] = torch.round(F.sigmoid(x_post))
                    else:
                        outs[:, i] = x_post.argmax(dim=-1)
                else:
                    outs[:, i] = x_post.flatten()
            else:
                outs.add_feature(x_post, feature)

        return outs

    def decode(self, x: Tensor) -> Tensor:
        return self(x)


class TemporalEmbedderDecoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_size: int,
        features: FeatureList,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.features = features
        self.mlp = MLPBlock(embedding_dim, hidden_size)
        self.feature_layers = nn.ModuleDict()

        for feature in features:
            if isinstance(feature, ContinuousFeature):
                self.feature_layers[feature.name] = nn.Sequential(
                    nn.Linear(hidden_size, 1),
                    nn.Sigmoid(),
                )
            elif isinstance(feature, CategoricalFeature):
                if len(feature) == 2:
                    self.feature_layers[feature.name] = nn.Sequential(
                        nn.Linear(hidden_size, 1),
                    )
                else:
                    self.feature_layers[feature.name] = nn.Sequential(
                        nn.Linear(hidden_size, len(feature)),
                    )

    def forward(
        self, x: Tensor, transform_and_unify_output: bool = False
    ) -> Union["DecoderOutput", Tensor]:
        x = self.mlp(x)

        if transform_and_unify_output:
            outs = torch.zeros((x.size(0), x.size(1), len(self.features))).type_as(x)
        else:
            outs = DecoderOutput()

        for i, feature in enumerate(self.features):
            x_post = self.feature_layers[feature.name](x)

            if transform_and_unify_output:
                if isinstance(feature, CategoricalFeature):
                    if len(feature) == 2:
                        outs[:, :, i] = torch.round(F.sigmoid(x_post))
                    else:
                        outs[:, :, i] = x_post.argmax(dim=-1)
                else:
                    outs[:, :, i] = x_post.flatten(start_dim=1)
            else:
                outs.add_feature(x_post, feature)

        return outs

    def decode(self, x: Tensor) -> Tensor:
        return self(x)


def frange_cycle_linear(n_iter, start=0.0, stop=1.0, n_cycle=4, ratio=0.5):
    # adapted from https://github.com/haofuml/cyclical_annealing/tree/master
    L = torch.ones(n_iter) * stop
    period = n_iter / n_cycle
    step = (stop - start) / (period * ratio)  # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i + c * period) < n_iter):
            L[int(i + c * period)] = v
            v += step
            i += 1
    return L


class PatientEmbeddedAE(LightningModule):
    def __init__(
        self,
        features: FeatureList,
        embedding_dim: int,
        hidden_size: int,
        continuous_encoding_freq_length: int,
        transformer_encoder_n_heads: int,
        transformer_encoder_dim_forward: int,
        num_transformer_encoder_layers: int,
        num_channels: int,
        static_pooling: Literal["mean", "max", "attn"] = "attn",
        rnn_type: Literal["lstm", "gru"] = "gru",
        learning_rate: float = 1e-3,
        dropout_p: float = 0.1,
        variational: bool = False,
        variational_beta_weight: Optional[float] = None,
        variational_beta_schedule: Optional[
            Union[Tensor, Literal["decreasing"]]
        ] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.static_encoder = StaticEmbeddedEncoder(
            features.static_features(),
            transformer_encoder_n_heads,
            transformer_encoder_dim_forward,
            embedding_dim,
            continuous_encoding_freq_length,
            pooling=static_pooling,
            dropout_p=dropout_p,
            variational=variational,
        )
        self.static_decoder = StaticEmbedderDecoder(
            embedding_dim, hidden_size, features.static_features()
        )

        self.temporal_encoder = TemporalEmbedderEncoder(
            features.temporal_features(),
            hidden_size,
            embedding_dim,
            num_channels,
            transformer_encoder_n_heads,
            transformer_encoder_dim_forward,
            num_transformer_encoder_layers,
            continuous_encoding_freq_length,
            rnn_type=rnn_type,
            dropout_p=dropout_p,
            variational=variational,
        )
        self.temporal_decoder = TemporalEmbedderDecoder(
            embedding_dim, hidden_size, features.temporal_features()
        )

        if self.hparams["variational_beta_schedule"] == "decreasing":
            self.best_kl = float("inf")
            self.beta_max = 0.01
            self.beta_min = 1e-5
            self.beta = self.beta_max
            self.beta_patience = 0

    def encode(self, x_s: Tensor, x_t: Tensor) -> Tuple[Tensor, Tensor]:
        static_enc = self.static_encoder(x_s)
        temp_enc = self.temporal_encoder(x_t)
        return static_enc, temp_enc

    def decode(
        self,
        h_s: Tensor,
        h_t: Tensor,
        transform_and_unify_output: bool = False,
        adjust_to_seq_lengths: Optional[Tensor] = None,
    ) -> Union[Tuple[Tensor, Tensor], "DecoderOutput"]:
        x_s = self.static_decoder(
            h_s, transform_and_unify_output=transform_and_unify_output
        )
        x_t = self.temporal_decoder(
            h_t, transform_and_unify_output=transform_and_unify_output
        )

        if not transform_and_unify_output and adjust_to_seq_lengths is not None:
            raise ValueError(
                "`transform_and_unify_output` must be True if `adjust_to_seq_lengths` is provided"
            )

        if transform_and_unify_output and adjust_to_seq_lengths is not None:
            for i, seq_length in enumerate(adjust_to_seq_lengths):
                x_t[i, seq_length:] = 0

        return x_s, x_t

    def forward(
        self, x_s: Tensor, x_t: Tensor, transform_and_unify_output: bool = False
    ) -> Dict[str, Union[Tensor, "DecoderOutput"]]:
        if self.hparams["variational"]:
            static_enc, temp_enc = self.encode(x_s, x_t)
            h_s, static_mu, static_logvar = static_enc
            h_t, temp_mu, temp_logvar = temp_enc
        else:
            h_s, h_t = self.encode(x_s, x_t)

        x_s_tilde, x_t_tilde = self.decode(
            h_s,
            h_t,
            transform_and_unify_output=transform_and_unify_output,
        )

        out = {"h_s": h_s, "h_t": h_t, "x_s_tilde": x_s_tilde, "x_t_tilde": x_t_tilde}

        if self.hparams["variational"]:
            out["static_mu"] = static_mu
            out["static_logvar"] = static_logvar
            out["temp_mu"] = temp_mu
            out["temp_logvar"] = temp_logvar

        return out

    def calculate_binary_loss(self, x: Tensor, outputs: "DecoderOutput") -> Tensor:
        x_s_binary = []
        x_s_tilde_binary = []

        for i, feature, values in outputs.get_binary():
            if x.dim() == 2:
                x_s_binary.append(x[:, i])
            else:
                x_s_binary.append(x[:, :, i].flatten())
            x_s_tilde_binary.append(values.flatten())

        if not x_s_tilde_binary:
            return torch.tensor(0.0)

        return F.binary_cross_entropy(
            torch.concat(x_s_tilde_binary), torch.concat(x_s_binary)
        )

    def calculate_categorical_loss(self, x: Tensor, outputs: "DecoderOutput") -> Tensor:
        categorical_losses = []

        for i, feature, values in outputs.get_categorical():
            if x.dim() == 2:
                categorical_losses.append(F.cross_entropy(values, x[:, i].long()))
            else:
                categorical_losses.append(
                    F.cross_entropy(
                        values.view(values.size(0) * values.size(1), -1),
                        x[:, :, i].view(values.size(0) * values.size(1)).long(),
                    )
                )

        if not categorical_losses:
            return torch.tensor(0.0)

        return torch.stack(categorical_losses).mean()

    def calculate_continuous_loss(self, x: Tensor, outputs: "DecoderOutput") -> Tensor:
        x_s_continuous = []
        x_s_tilde_continuous = []

        for i, feature, values in outputs.get_continuous():
            if x.dim() == 2:
                x_s_continuous.append(x[:, i])
            else:
                x_s_continuous.append(x[:, :, i].flatten())
            x_s_tilde_continuous.append(values.flatten())

        if not x_s_tilde_continuous:
            return torch.tensor(0.0)

        return F.mse_loss(
            torch.concat(x_s_tilde_continuous), torch.concat(x_s_continuous)
        )

    def calculate_kl_loss(
        self,
        static_mu: Tensor,
        static_logvar: Tensor,
        temp_mu: Tensor,
        temp_logvar: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        static_kl_loss = -0.5 * torch.sum(
            1 + static_logvar - static_mu.pow(2) - static_logvar.exp()
        )
        temp = 1 + temp_logvar - temp_mu.pow(2) - temp_logvar.exp()
        temp_kl_loss = -0.5 * torch.mean(temp.mean(-1).mean())

        return static_kl_loss, temp_kl_loss

    def calculate_loss(
        self, x_s: Tensor, x_t: Tensor, outputs: Dict[str, "DecoderOutput"]
    ) -> Tensor:
        loss_binary_s = self.calculate_binary_loss(x_s, outputs["x_s_tilde"])
        loss_categorical_s = self.calculate_categorical_loss(x_s, outputs["x_s_tilde"])
        loss_continuous_s = self.calculate_continuous_loss(x_s, outputs["x_s_tilde"])

        loss_binary_t = self.calculate_binary_loss(x_t, outputs["x_t_tilde"])
        loss_categorical_t = self.calculate_categorical_loss(x_t, outputs["x_t_tilde"])
        loss_continuous_t = self.calculate_continuous_loss(x_t, outputs["x_t_tilde"])

        loss = (
            loss_binary_s
            + loss_categorical_s
            + loss_continuous_s
            + loss_binary_t
            + loss_categorical_t
            + loss_continuous_t
        )

        losses = {
            "loss": loss,
            "reconstruction_loss": loss,
            "reconstruction_loss_s": loss_binary_s
            + loss_categorical_s
            + loss_continuous_s,
            "reconstruction_loss_t": loss_binary_t
            + loss_categorical_t
            + loss_continuous_t,
        }

        if self.hparams["variational"]:
            static_kl_loss, temp_kl_loss = self.calculate_kl_loss(
                outputs["static_mu"],
                outputs["static_logvar"],
                outputs["temp_mu"],
                outputs["temp_logvar"],
            )
            if self.hparams["variational_beta_schedule"] is not None:
                if (
                    isinstance(self.hparams["variational_beta_schedule"], str)
                    and self.hparams["variational_beta_schedule"] == "decreasing"
                ):
                    variational_beta_weight = self.beta
                    if losses["loss"] < self.best_kl:
                        self.best_kl = losses["loss"]
                        self.beta_patience = 0
                    else:
                        self.beta_patience += 1

                        if self.beta_patience == 10:
                            if self.beta > self.beta_min:
                                self.beta *= 0.7
                else:
                    variational_beta_weight = self.hparams["variational_beta_schedule"][
                        min(
                            self.global_step,
                            len(self.hparams["variational_beta_schedule"]) - 1,
                        )
                    ]
            else:
                variational_beta_weight = self.hparams["variational_beta_weight"]

            losses["kl_loss"] = variational_beta_weight * (
                static_kl_loss + temp_kl_loss
            )
            losses["kl_static_loss"] = variational_beta_weight * static_kl_loss
            losses["kl_temp_loss"] = variational_beta_weight * temp_kl_loss
            losses["loss"] += losses["kl_loss"]

        return losses

    def training_step(
        self, batch: Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]
    ) -> Tensor:
        x_s, x_t, _ = batch
        outputs = self(
            x_s,
            x_t,
        )

        loss_dict = self.calculate_loss(x_s, x_t, outputs)
        self.log_dict(loss_dict, prog_bar=True)

        return loss_dict["loss"]

    def validation_step(self, batch, _) -> None:
        x_s, x_t, _ = batch

        outputs = self(x_s, x_t)
        loss_dict = self.calculate_loss(x_s, x_t, outputs)

        for k in list(loss_dict.keys()):
            loss_dict[f"val_{k}"] = loss_dict.pop(k)

        self.log_dict(loss_dict, prog_bar=True)

    def configure_optimizers(self) -> Optimizer:
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams["learning_rate"],
        )

        return opt
