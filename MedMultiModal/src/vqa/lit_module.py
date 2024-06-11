"""VQA pipeline with attention-based fusion and optional MEVF.

The fusion module is called Bilinear Attention Network (BAN) [1]; the source code of
which is accessible in [2]. Mixture of Enhanced Visual Features (MEVF) [3, 4] enhances
the performance of medical question answering models.

References
----------
[1] Kim, Jin-Hwa, Jaehyun Jun, and Byoung-Tak Zhang. "Bilinear attention networks."
    Advances in neural information processing systems 31 (2018).
[2] https://github.com/jnhwkim/ban-vqa
[3] Nguyen, Binh D., Thanh-Toan Do, Binh X. Nguyen, Tuong Do, Erman Tjiputra,
    and Quang D. Tran, "Overcoming data limitation in medical visual question
    answering." In Medical Image Computing and Computer Assisted Intervention,
    MICCAI 2019: 22nd International Conference, 2019.
[4] https://github.com/aioz-ai/MICCAI19-MedVQA
"""

import json
import os
from typing import Any, Dict, Literal, Tuple, Union

import lightning as L  # noqa: N812
import torch
import wandb
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig
from vqa import base_model


class VQA(L.LightningModule):
    """Visual question answering pipeline with attention-based fusion and optional MEVF.

    Parameters
    ----------
    cfg: DictConfig
        Configuration of all pipeline components.
        It contains the following keys.
        `encoder`:
            Visual encoder module configs.
        `checkpoint`:
            Checkpoint of the visual encoder to load.
        `autoencoder`:
            Auto-encoder module configs.
        `fusion`:
            Attention-based fusion module configs.
        `classifier`:
            Classification head module configs.
        `lr_scheduler`:
            Configuration of the learning rate scheduler.
        `dump_result`: bool
            Whether or not to save results in json file.
    optimizer: DictConfig, optional
        Optimizer for the whole pipeline.
    lr_scheduler: DictConfig, optional
        Learning rate scheduler for the whole pipeline.
        This must be `None`, since an `lr_scheduler` is implemented inside the pipeline.
    """

    def __init__(
        self,
        network: DictConfig,
        optimizer: DictConfig,
    ) -> None:
        """Initialize the pipeline."""
        super().__init__()

        self.net_cfg = network

        # create VQA model
        constructor = f"build_{network.fusion.arch.lower()}"
        self.pipeline = getattr(base_model, constructor)(network)

        # loss function
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
        self.ae_criterion = torch.nn.MSELoss()

        # optimizer and lr_scheduler
        self._optim_cfg = optimizer

        # scheduler learning rate
        self._lr_default = self._optim_cfg.lr
        self._lr_decay_step = 2
        self._lr_decay_rate = 0.75
        self._lr_decay_epochs = range(10, 20, self._lr_decay_step)
        self._gradual_warmup_steps = [
            0.5 * self._lr_default,
            1.0 * self._lr_default,
            1.5 * self._lr_default,
            2.0 * self._lr_default,
        ]
        self._epoch = 0

        self.keys = ["count", "real", "true", "real_percent", "score", "score_percent"]

    def forward(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Move a batch thought the pipeline."""
        return self.pipeline(batch)

    def on_train_epoch_start(self) -> None:
        """Manually schedule the learning rate."""
        if self._epoch < len(self._gradual_warmup_steps):
            self.optimizers().param_groups[0]["lr"] = self._gradual_warmup_steps[
                self._epoch
            ]
        elif self._epoch in self._lr_decay_epochs:
            self.optimizers().param_groups[0]["lr"] *= self._lr_decay_rate

    def on_train_epoch_end(self) -> None:
        """Take required step for lr_scheduler."""
        self._epoch += 1

    def compute_scores_with_logits(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute accuracy score."""
        logits = torch.max(logits, 1)[1].detach()  # argmax
        one_hots = torch.zeros_like(labels)
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        return one_hots * labels

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """Compute loss for the batch.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch of data. It should contain the keys `"rgb"` and
            `"text"`.
        batch_idx : int
            The index of the batch.

        Returns
        -------
        torch.Tensor
            The loss for the batch.
        """
        answers = batch["rgb_target"]  # batch x num_classes
        if self.net_cfg.autoencoder.enabled:
            ae_img_data = batch["rgb"][1]  # batch x num_objs x H x W

        # MEVF loss computation: classification loss
        if self.net_cfg.autoencoder.enabled:
            features, decoder = self.pipeline(batch)
        else:
            features = self.pipeline(batch)

        preds = self.pipeline.classifier(features)  # batch x num_classes
        loss = self.criterion(preds.float(), answers)

        # MEVF loss computation: reconstruction loss
        if self.net_cfg.autoencoder.enabled:
            loss_ae = self.ae_criterion(ae_img_data, decoder)
            # multi-task loss
            loss = loss + (loss_ae * self.net_cfg.autoencoder.alpha)
        loss = loss / answers.size()[0]

        self.log(
            "train/loss",
            loss if not self.fabric else self.all_gather(loss.clone().detach()).mean(),
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def on_validation_epoch_start(self) -> None:
        """Set up variables for accumulating validation statistics."""
        return self._on_eval_epoch_start()

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._eval_step(batch, batch_idx, "val")

    def on_validation_epoch_end(self) -> None:
        return self._on_eval_epoch_end("val")

    def on_test_epoch_start(self) -> None:
        """Set up variables for accumulating test statistics."""
        return self._on_eval_epoch_start()

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._eval_step(batch, batch_idx, "test")

    def on_test_epoch_end(self) -> None:
        return self._on_eval_epoch_end("test")

    def _on_eval_epoch_start(self) -> None:
        """Define some constants and state-keeping variables."""
        self.question_types_result = {}
        self.result = {}

    def _eval_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        eval_type: Literal["val", "test"],
    ) -> torch.Tensor:
        """Compute the test metrics for the batch.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            The batch of data. It should contain the keys `"rgb"` and `"text"`.
        batch_idx : int
            The index of the batch.
        eval_type : {"validation", "test"}
            The type of evaluation being performed.

        Returns
        -------
        Dict[str, torch.Tensor]
            A map of names metrics to their values.
        """
        assert eval_type in ["val", "test"]

        answers = batch["rgb_target"]  # batch x num_classes
        if self.net_cfg.autoencoder.enabled:
            ae_img_data = batch["rgb"][1]  # batch x num_objs x H x W

        # MEVF loss computation: classification loss
        if self.net_cfg.autoencoder.enabled:
            features, decoder = self.pipeline(batch)
        else:
            features = self.pipeline(batch)

        preds = self.pipeline.classifier(features)  # batch x num_classes
        loss = self.criterion(preds.float(), answers)

        # MEVF loss computation: reconstruction loss
        if self.net_cfg.autoencoder.enabled:
            loss_ae = self.ae_criterion(ae_img_data, decoder)
            # multi-task loss
            loss = loss + (loss_ae * self.net_cfg.autoencoder.alpha)
        loss = loss / answers.size()[0]

        # compute batch_score
        final_preds = preds
        batch_score = self.compute_scores_with_logits(
            final_preds, answers.detach()
        ).sum()

        # Compute accuracy for each type answer
        answer_type = batch["answer_type"]
        batch["question_type"] = batch["question_type"][0].split(", ")
        for i in answer_type:
            self.result.setdefault(i, {j: 0.0 for j in self.keys})["count"] += (
                preds.size()[0]
            )
            self.result.setdefault(i, {j: 0.0 for j in self.keys})["true"] += float(
                batch_score
            )
            self.result.setdefault(i, {j: 0.0 for j in self.keys})["real"] += float(
                batch["rgb_target"].sum()
            )

            self.result.setdefault("ALL", {j: 0.0 for j in self.keys})["count"] += (
                preds.size()[0]
            )
            self.result.setdefault("ALL", {j: 0.0 for j in self.keys})["true"] += float(
                batch_score
            )
            self.result.setdefault("ALL", {j: 0.0 for j in self.keys})["real"] += float(
                batch["rgb_target"].sum()
            )

            for j in batch["question_type"]:
                self.question_types_result.setdefault(
                    i,
                    {},
                ).setdefault(j, {m: 0.0 for m in self.keys})["count"] += preds.size()[0]
                self.question_types_result.setdefault(
                    i,
                    {},
                ).setdefault(j, {m: 0.0 for m in self.keys})["true"] += float(
                    batch_score
                )
                self.question_types_result.setdefault(
                    i,
                    {},
                ).setdefault(j, {m: 0.0 for m in self.keys})["real"] += float(
                    batch["rgb_target"].sum()
                )

                self.question_types_result.setdefault(
                    "ALL",
                    {},
                ).setdefault(j, {m: 0.0 for m in self.keys})["count"] += preds.size()[0]
                self.question_types_result.setdefault(
                    "ALL",
                    {},
                ).setdefault(j, {m: 0.0 for m in self.keys})["true"] += float(
                    batch_score
                )
                self.question_types_result.setdefault(
                    "ALL",
                    {},
                ).setdefault(j, {m: 0.0 for m in self.keys})["real"] += float(
                    batch["rgb_target"].sum()
                )

        self.log(
            f"{eval_type}/loss",
            loss if not self.fabric else self.all_gather(loss.clone().detach()).mean(),
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def _on_eval_epoch_end(self, eval_type: Literal["val", "test"]) -> None:
        """Compute the test metrics for the entire test set."""
        # gather metric values from all processes
        self._gather_results()

        # compute metrics for the whole set
        for i in self.result:
            self.result[i]["score"] = self.result[i]["true"] / self.result[i]["count"]
            self.result[i]["score_percent"] = self.result[i]["score"] * 100
            self.result[i]["real_percent"] = (
                self.result[i]["real"] / self.result[i]["count"] * 100
            )
            for j in self.question_types_result[i]:
                if self.question_types_result[i][j]["count"] != 0.0:
                    self.question_types_result[i][j]["score"] = (
                        self.question_types_result[i][j]["true"]
                        / self.question_types_result[i][j]["count"]
                    )
                    self.question_types_result[i][j]["score_percent"] = (
                        self.question_types_result[i][j]["score"] * 100
                    )
                    if self.question_types_result[i][j]["real"] != 0.0:
                        self.question_types_result[i][j]["real_percent"] = (
                            self.question_types_result[i][j]["real"]
                            / self.question_types_result[i][j]["count"]
                            * 100.0
                        )

        # flatten the results and log
        result = {}
        for ki in self.result:
            for kj in self.keys:
                if "score_percent" in kj:
                    result[f"{eval_type}/{ki.lower().replace('/', '_')}_{kj}"] = (
                        self.result[ki][kj]
                    )
        self.log_dict(result)

        # dump results to json file
        if self.net_cfg.dump_result and rank_zero_only.rank == 0:
            if not os.path.isdir("./medvqa_output/"):
                os.mkdir("./medvqa_output/")
            with open("medvqa_output/results.json", "w", encoding="utf-8") as file:
                json.dump(self.result, file, indent=4)
                wandb.save("medvqa_output/results.json")
            with open(
                "medvqa_output/results_detail.json", "w", encoding="utf-8"
            ) as file:
                json.dump(self.question_types_result, file, indent=4)
                wandb.save("medvqa_output/results_detail.json")

        if not self.net_cfg.dump_result:
            self.print(self.result)
            self.print(self.question_types_result)

    def _gather_results(self) -> None:
        """Gather metric values from all processes."""
        for k1, _ in self.result.items():
            for k2, _ in self.result[k1].items():
                self.result[k1][k2] = float(self.all_gather(self.result[k1][k2]).sum())

        for k1, _ in self.question_types_result.items():
            for k2, _ in self.question_types_result[k1].items():
                for k3, _ in self.question_types_result[k1][k2].items():
                    self.question_types_result[k1][k2][k3] = float(
                        self.all_gather(self.question_types_result[k1][k2][k3]).sum()
                    )

    def configure_optimizers(
        self,
    ) -> Union[
        torch.optim.Optimizer,
        Dict[str, Union[torch.optim.Optimizer, Dict[str, Any]]],
    ]:
        """Configure the optimizer and learning rate scheduler."""
        # freeze parameters that are not used in producing the loss
        self.pipeline.q_prj[0].main[
            1
        ].parametrizations.weight.original0.requires_grad = False
        self.pipeline.q_prj[0].main[
            1
        ].parametrizations.weight.original1.requires_grad = False
        self.pipeline.q_prj[0].main[1].bias.requires_grad = False
        self.pipeline.b_net[0].q_net.main[
            1
        ].parametrizations.weight.original0.requires_grad = False
        self.pipeline.b_net[0].q_net.main[
            1
        ].parametrizations.weight.original1.requires_grad = False
        self.pipeline.b_net[0].q_net.main[1].bias.requires_grad = False
        self.pipeline.b_net[0].v_net.main[
            1
        ].parametrizations.weight.original0.requires_grad = False
        self.pipeline.b_net[0].v_net.main[
            1
        ].parametrizations.weight.original1.requires_grad = False
        self.pipeline.b_net[0].v_net.main[1].bias.requires_grad = False

        optimizer = torch.optim.Adamax(
            filter(lambda p: p.requires_grad, self.pipeline.parameters()),
            self._optim_cfg.lr if self._optim_cfg is not None else 1e-3,
        )

        return optimizer
