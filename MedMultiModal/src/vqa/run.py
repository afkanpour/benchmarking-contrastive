"""Configurable script for training or evaluation."""

import os
from typing import Optional, Tuple

import hydra
import lightning as L  # noqa: N812
from lightning.pytorch.trainer import Trainer
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from vqa.lit_module import VQA

OmegaConf.register_new_resolver(
    name="get_obj", resolver=lambda obj: hydra.utils.get_object(obj)
)


def instantiate_datasets(
    dataset_cfg: DictConfig,
) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
    """Instantiate datasets from config.

    Parameters
    ----------
    dataset_cfg : DictConfig
        A DictConfig object containing dataset configurations.

    Returns
    -------
    List
        A list of instantiated datasets.
    """
    if not dataset_cfg:  # dataset is a required field in the config
        raise ValueError("No dataset configs found!")

    if not isinstance(dataset_cfg, DictConfig):  # single dataset
        raise TypeError("Dataset config must be a DictConfig!")

    def _instantiate_split(split_cfg: DictConfig, split: str) -> Optional[Dataset]:
        if isinstance(split_cfg, DictConfig) and "_target_" in split_cfg:
            dataset = None
            if split == "train":
                dataset = hydra.utils.instantiate(split_cfg, _convert_="partial")
            elif split == "valid":
                dataset = hydra.utils.instantiate(split_cfg, _convert_="partial")
            elif split == "test":
                dataset = hydra.utils.instantiate(split_cfg, _convert_="partial")

            return dataset
        else:
            raise ValueError(
                f"Invalid dataset config for split '{split}'! Must be a DictConfig with '_target_' key."
            )

    train_dataset, valid_dataset, test_dataset = None, None, None
    if "train" in dataset_cfg:
        train_dataset = _instantiate_split(dataset_cfg.train, "train")
    elif "valid" in dataset_cfg:
        valid_dataset = _instantiate_split(dataset_cfg.valid, "valid")
    elif "test" in dataset_cfg:
        test_dataset = _instantiate_split(dataset_cfg.test, "test")

    return train_dataset, valid_dataset, test_dataset


@hydra.main(config_path="./configs", config_name="default", version_base="1.3")  # type: ignore[misc]
def main(cfg: DictConfig) -> None:  # noqa: PLR0912
    """Training entry point."""
    rand_seed: Optional[int] = cfg.get("seed", None)
    if rand_seed is not None:
        L.seed_everything(rand_seed, workers=True)

    out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # configure trainer
    trainer = Trainer(
        devices=1,
        num_nodes=1,
        benchmark=False,
        max_epochs=20,
        precision="32-true",
        deterministic=True,
        gradient_clip_val=0.25,
        gradient_clip_algorithm="norm",
        enable_checkpointing=True,
        enable_model_summary=True,
        log_every_n_steps=50,
        default_root_dir=os.path.join(out_dir, "checkpoints"),
        # loggers=[
        #     WandbLogger(
        #         name=cfg.experiment_name,
        #         project=os.env.get("WANDB_PROJECT", "vqa"),
        #         save_dir=out_dir,
        #         resume="allow",
        #         job_type=cfg.job_type,
        #         tags=cfg.tags,
        #         config=OmegaConf.to_object(cfg),
        #     )
        # ],
    )

    train_dataset, val_dataset, test_dataset = instantiate_datasets(cfg.dataset)

    # prepare dataloaders
    if cfg.job_type == "train":
        train_loader = DataLoader(
            train_dataset, batch_size=32, shuffle=True, drop_last=True, pin_memory=True
        )
        val_loader: Optional[DataLoader] = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
            )
    else:
        test_loader: Optional[DataLoader] = None
        if test_dataset is not None:
            test_loader = DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
            )

    # create lightning module
    if cfg.get("module") is None:
        raise ValueError("The `module` field is required in the configuration file.")
    model: L.LightningModule = VQA(cfg.module.network, cfg.module.optimizer)

    trainer.print(OmegaConf.to_yaml(cfg, resolve=True))

    if cfg.job_type == "train":
        # train model
        trainer.fit(
            model, train_loader, val_loader, ckpt_path=cfg.get("resume_from_checkpoint")
        )
    elif cfg.job_type == "eval":
        # evaluate model
        trainer.test(model, test_loader, ckpt_path=cfg.get("resume_from_checkpoint"))
    else:
        raise ValueError(
            "Expected `job_type` to be one of ['train', 'eval'], but got "
            f"{cfg.job_type}."
        )


if __name__ == "__main__":
    main()
