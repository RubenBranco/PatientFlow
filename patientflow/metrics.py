import logging
from multiprocessing import cpu_count
from typing import Literal, Optional, Tuple, Union

import torch
from lightning.pytorch import (
    LightningModule,
    Trainer,
    seed_everything,
)
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import AUROC, Accuracy


class DetectorModule(nn.Module):
    def __init__(
        self,
        num_temporal_features: int,
        temporal_hidden_size: int,
        pack_input: bool = True,
        rnn_type: Literal["gru", "lstm"] = "gru",
        num_static_features: Optional[int] = None,
        static_hidden_size: Optional[int] = None,
    ):
        super().__init__()
        self.num_temporal_features = num_temporal_features
        self.temporal_hidden_size = temporal_hidden_size
        self.pack_input = pack_input
        self.rnn_type = rnn_type
        self.num_static_features = num_static_features
        self.static_hidden_size = static_hidden_size

        if num_static_features is not None:
            self.static_encoder = nn.Sequential(
                nn.Linear(num_static_features, static_hidden_size),
                nn.LeakyReLU(),
            )

        rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU

        self.rnn = rnn_cls(
            input_size=num_temporal_features,
            hidden_size=temporal_hidden_size,
            batch_first=True,
        )
        self.fc = nn.Linear(
            temporal_hidden_size + static_hidden_size
            if num_static_features is not None
            else temporal_hidden_size,
            1,
        )

    def forward(
        self,
        x_t: Tensor,
        seq_lengths: Optional[Tensor] = None,
        x_s: Optional[Tensor] = None,
    ) -> Tensor:
        if self.pack_input and seq_lengths is not None:
            x_t = nn.utils.rnn.pack_padded_sequence(  # type: ignore
                x_t,
                lengths=seq_lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )

        _, h_n = self.rnn(x_t)

        if self.rnn_type == "lstm":
            h_n = h_n[0]

        h_n = nn.functional.relu(h_n[-1])

        if self.num_static_features is not None:
            h_n = torch.cat([h_n, self.static_encoder(x_s)], dim=1)

        return self.fc(h_n)


class PrognosticNetwork(LightningModule):
    def __init__(
        self,
        num_temporal_features: int,
        temporal_hidden_size: int = 256,
        pack_input: bool = True,
        rnn_type: Literal["gru", "lstm"] = "gru",
        static_hidden_size: int = 256,
        num_static_features: Optional[int] = None,
        eval_metric: Literal["auroc", "acc"] = "auroc",
        threshold: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.prog_net = DetectorModule(
            num_temporal_features,
            temporal_hidden_size,
            pack_input,
            rnn_type,
            num_static_features,
            static_hidden_size,
        )
        self._has_static_data = self.hparams["num_static_features"] is not None

        if eval_metric == "auroc":
            self.metric = AUROC(task="binary")
        elif eval_metric == "acc":
            self.metric = Accuracy(task="binary")
        else:
            raise ValueError(f"Invalid eval_metric: {eval_metric}")

    def forward(
        self,
        x_t: Tensor,
        seq_lengths: Optional[Tensor] = None,
        x_s: Optional[Tensor] = None,
        argmax: bool = False,
    ) -> Tensor:
        y_pred = self.prog_net(x_t, seq_lengths, x_s)

        if argmax:
            y_pred = y_pred.argmax(dim=1)

        return y_pred

    def training_step(
        self,
        batch: Union[
            Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]
        ],
    ) -> Tensor:
        if self._has_static_data:
            x_s, x_t, seq_lengths, y = batch
        else:
            x_t, seq_lengths, y = batch
            x_s = None

        y_pred = self.prog_net(x_t, seq_lengths, x_s)
        loss = nn.functional.binary_cross_entropy_with_logits(y_pred, y.unsqueeze(-1))

        return loss

    def test_step(
        self,
        batch: Union[
            Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]
        ],
        _,
    ) -> None:
        if self._has_static_data:
            x_s, x_t, seq_lengths, y = batch
        else:
            x_t, seq_lengths, y = batch
            x_s = None

        y_pred = self.prog_net(x_t, seq_lengths, x_s)

        self.metric.update(
            (nn.functional.sigmoid(y_pred) >= self.hparams["threshold"]).float(), y
        )

    def on_test_epoch_end(self) -> None:
        self.log_dict({self.hparams["eval_metric"]: self.metric.compute()})
        self.metric.reset()

    def configure_optimizers(self) -> Optimizer:
        return torch.optim.Adam(self.parameters(), lr=1e-4)


class PrognosisMetric:
    """
    Based on SDV Metric API and DetectionMetric
    """

    name = "Detection Metric"
    min_value = 0.0
    max_value = 1.0

    @staticmethod
    def compute(
        train_ds: TensorDataset,
        test_ds: TensorDataset,
        seed: int = 42,
        batch_size: int = 64,
        accelerator: Literal["cpu", "gpu"] = "gpu",
        rnn_type: Literal["gru", "lstm"] = "gru",
        static_hidden_size: int = 256,
        temporal_hidden_size: int = 256,
        eval_metric: Literal["auroc", "acc"] = "auroc",
        enable_progress_bar: bool = False,
        epochs: int = 1024,
        data_loader_num_workers: Optional[int] = None,
        data_loader_context: Optional[str] = None,
    ) -> float:
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        logging.getLogger("lightning.fabric.utilities.seed").setLevel(logging.ERROR)
        logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.ERROR)
        logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(
            logging.ERROR
        )
        logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.ERROR)

        seed_everything(seed)

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=cpu_count()
            if data_loader_num_workers is None
            else data_loader_num_workers,
            persistent_workers=True,
            multiprocessing_context="spawn"
            if data_loader_context is None
            else data_loader_context,
        )
        test_dl = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=cpu_count()
            if data_loader_num_workers is None
            else data_loader_num_workers,
            persistent_workers=True,
            multiprocessing_context="spawn"
            if data_loader_context is None
            else data_loader_context,
        )

        model = PrognosticNetwork(
            train_ds[0][1].size(-1),
            num_static_features=train_ds[0][0].size(-1),
            static_hidden_size=static_hidden_size,
            temporal_hidden_size=temporal_hidden_size,
            eval_metric=eval_metric,
            rnn_type=rnn_type,
        )
        trainer = Trainer(
            max_epochs=epochs,
            enable_checkpointing=False,
            logger=False,
            accelerator=accelerator,
            enable_progress_bar=enable_progress_bar,
            enable_model_summary=False,
        )
        trainer.fit(model, train_dl)
        test_metric = trainer.test(model, test_dl, verbose=False)[0][eval_metric]
        del model, trainer, train_ds, test_ds, train_dl, test_dl  # clean up
        logging.getLogger("pytorch_lightning").setLevel(logging.INFO)
        logging.getLogger("lightning.fabric.utilities.seed").setLevel(logging.INFO)
        logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.INFO)
        logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(
            logging.INFO
        )
        logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.INFO)

        return test_metric
