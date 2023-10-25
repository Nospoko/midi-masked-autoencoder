import os

import hydra
import torch
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from omegaconf import OmegaConf
from huggingface_hub import upload_file
from torch.utils.data import Subset, DataLoader
from datasets import load_dataset, concatenate_datasets

import wandb
from data.dataset import MidiDataset
from models.mae import MidiMaskedAutoencoder


def makedir_if_not_exists(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)


def preprocess_dataset(
    dataset_name: list[str],
    batch_size: int,
    num_workers: int,
    pitch_shift_probability: float,
    time_stretch_probability: float,
    *,
    overfit_single_batch: bool = False,
):
    hf_token = os.environ["HUGGINGFACE_TOKEN"]

    train_ds = []
    val_ds = []
    test_ds = []

    for ds_name in dataset_name:
        tr_ds = load_dataset(ds_name, split="train", use_auth_token=hf_token)
        v_ds = load_dataset(ds_name, split="validation", use_auth_token=hf_token)
        t_ds = load_dataset(ds_name, split="test", use_auth_token=hf_token)

        train_ds.append(tr_ds)
        val_ds.append(v_ds)
        test_ds.append(t_ds)

    train_ds = concatenate_datasets(train_ds)
    val_ds = concatenate_datasets(val_ds)
    test_ds = concatenate_datasets(test_ds)

    train_ds = MidiDataset(
        train_ds,
        pitch_shift_probability=pitch_shift_probability,
        time_stretch_probability=time_stretch_probability,
    )
    val_ds = MidiDataset(val_ds, pitch_shift_probability=0.0, time_stretch_probability=0.0)
    test_ds = MidiDataset(test_ds, pitch_shift_probability=0.0, time_stretch_probability=0.0)

    if overfit_single_batch:
        train_ds = Subset(train_ds, indices=range(batch_size))
        val_ds = Subset(val_ds, indices=range(batch_size))
        test_ds = Subset(test_ds, indices=range(batch_size))

    # dataloaders
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)

    return train_dataloader, val_dataloader, test_dataloader


def forward_step(
    model: MidiMaskedAutoencoder,
    batch: dict[str, torch.Tensor, torch.Tensor, torch.Tensor],
    masking_ratio: float,
    loss_lambdas: dict,
    device: torch.device,
):
    pitch = batch["pitch"].to(device)
    velocity = batch["velocity"].to(device)
    dstart = batch["dstart"].to(device)
    duration = batch["duration"].to(device)

    pred_pitch, pred_dynamics, mask = model(
        pitch=pitch,
        velocity=velocity,
        dstart=dstart,
        duration=duration,
        masking_ratio=masking_ratio,
    )
    # calculate losses
    pitch_loss = F.cross_entropy(pred_pitch, pitch, reduction="none")
    velocity_loss = F.mse_loss(pred_dynamics[:, :, 0], velocity, reduction="none")
    dstart_loss = F.mse_loss(pred_dynamics[:, :, 1], dstart, reduction="none")
    duration_loss = F.mse_loss(pred_dynamics[:, :, 2], duration, reduction="none")

    # normalize losses
    pitch_loss = loss_lambdas.pitch * (pitch_loss * mask).sum() / mask.sum()
    velocity_loss = loss_lambdas.velocity * (velocity_loss * mask).sum() / mask.sum()
    dstart_loss = loss_lambdas.dstart * (dstart_loss * mask).sum() / mask.sum()
    duration_loss = loss_lambdas.duration * (duration_loss * mask).sum() / mask.sum()

    loss = pitch_loss + velocity_loss + dstart_loss + duration_loss

    return loss, pitch_loss, velocity_loss, dstart_loss, duration_loss


@torch.no_grad()
def validation_epoch(
    model: MidiMaskedAutoencoder,
    dataloader: DataLoader,
    masking_ratio: float,
    loss_lambdas: dict,
    device: torch.device,
) -> dict:
    # val epoch
    val_loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
    loss_epoch = 0.0
    pitch_loss_epoch = 0.0
    velocity_loss_epoch = 0.0
    dstart_loss_epoch = 0.0
    duration_loss_epoch = 0.0

    for batch_idx, batch in val_loop:
        # metrics returns loss and additional metrics if specified in step function
        loss, pitch_loss, velocity_loss, dstart_loss, duration_loss = forward_step(
            model, batch, masking_ratio, loss_lambdas, device
        )

        val_loop.set_postfix(
            {
                "loss": loss.item(),
                "pitch_loss": pitch_loss.item(),
                "velocity_loss": velocity_loss.item(),
                "dstart_loss": dstart_loss.item(),
                "duration_loss": duration_loss.item(),
            }
        )

        loss_epoch += loss.item()
        pitch_loss_epoch += pitch_loss.item()
        velocity_loss_epoch += velocity_loss.item()
        dstart_loss_epoch += dstart_loss.item()
        duration_loss_epoch += duration_loss.item()

    metrics = {
        "loss_epoch": loss_epoch / len(dataloader),
        "pitch_loss": pitch_loss_epoch / len(dataloader),
        "velocity_loss": velocity_loss_epoch / len(dataloader),
        "dstart_loss": dstart_loss_epoch / len(dataloader),
        "duration_loss": duration_loss_epoch / len(dataloader),
    }
    return metrics


def save_checkpoint(model: MidiMaskedAutoencoder, optimizer: optim.Optimizer, cfg: OmegaConf, save_path: str):
    # saving models
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
        },
        f=save_path,
    )


def upload_to_huggingface(ckpt_save_path: str, cfg: OmegaConf):
    # get huggingface token from environment variables
    token = os.environ["HUGGINGFACE_TOKEN"]

    # upload model to hugging face
    upload_file(ckpt_save_path, path_in_repo=f"{cfg.logger.run_name}.ckpt", repo_id=cfg.paths.hf_repo_id, token=token)


@hydra.main(config_path="configs", config_name="config-default", version_base="1.3.2")
def train(cfg: OmegaConf):
    wandb.login()

    # create dir if they don't exist
    makedir_if_not_exists(cfg.paths.log_dir)
    makedir_if_not_exists(cfg.paths.save_ckpt_dir)

    # dataset
    train_dataloader, val_dataloader, _ = preprocess_dataset(
        dataset_name=cfg.train.dataset_name,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        pitch_shift_probability=cfg.train.pitch_shift_probability,
        time_stretch_probability=cfg.train.time_stretch_probability,
        overfit_single_batch=cfg.train.overfit_single_batch,
    )

    # validate on quantized maestro
    _, maestro_test, _ = preprocess_dataset(
        dataset_name=["JasiekKaczmarczyk/maestro-v1-sustain-masked"],
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        pitch_shift_probability=cfg.train.pitch_shift_probability,
        time_stretch_probability=cfg.train.time_stretch_probability,
        overfit_single_batch=cfg.train.overfit_single_batch,
    )

    # logger
    wandb.init(
        project="midi-masked-autoencoder",
        name=cfg.logger.run_name,
        dir=cfg.paths.log_dir,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    device = torch.device(cfg.train.device)
    masking_ratio = cfg.train.masking_ratio
    loss_lambdas = cfg.train.loss_lambdas

    # model
    model = MidiMaskedAutoencoder(
        encoder_dim=cfg.model.encoder_dim,
        encoder_depth=cfg.model.encoder_depth,
        encoder_num_heads=cfg.model.encoder_num_heads,
        decoder_dim=cfg.model.decoder_dim,
        decoder_depth=cfg.model.decoder_depth,
        decoder_num_heads=cfg.model.decoder_num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
    ).to(device)

    # setting up optimizer
    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    # load checkpoint if specified in cfg
    if cfg.paths.load_ckpt_path is not None:
        checkpoint = torch.load(cfg.paths.load_ckpt_path)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    # checkpoint save path
    num_params_millions = sum([p.numel() for p in model.parameters()]) / 1_000_000
    save_path = f"{cfg.paths.save_ckpt_dir}/{cfg.logger.run_name}-params-{num_params_millions:.2f}M.ckpt"

    # step counts for logging to wandb
    step_count = 0

    for epoch in range(cfg.train.num_epochs):
        # train epoch
        model.train()
        train_loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=False)
        loss_epoch = 0.0
        pitch_loss_epoch = 0.0
        velocity_loss_epoch = 0.0
        dstart_loss_epoch = 0.0
        duration_loss_epoch = 0.0

        for batch_idx, batch in train_loop:
            # metrics returns loss and additional metrics if specified in step function
            loss, pitch_loss, velocity_loss, dstart_loss, duration_loss = forward_step(
                model, batch, masking_ratio, loss_lambdas, device
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            stats = {
                "loss": loss.item(),
                "pitch_loss": pitch_loss.item(),
                "velocity_loss": velocity_loss.item(),
                "dstart_loss": dstart_loss.item(),
                "duration_loss": duration_loss.item(),
            }

            train_loop.set_postfix(stats)

            step_count += 1
            loss_epoch += loss.item()
            pitch_loss_epoch += pitch_loss.item()
            velocity_loss_epoch += velocity_loss.item()
            dstart_loss_epoch += dstart_loss.item()
            duration_loss_epoch += duration_loss.item()

            if (batch_idx + 1) % cfg.logger.log_every_n_steps == 0:
                stats = {"train/" + key: value for key, value in stats.items()}

                # log metrics
                wandb.log(stats, step=step_count)

                # save model and optimizer states
                save_checkpoint(model, optimizer, cfg, save_path=save_path)

        training_metrics = {
            "train/loss_epoch": loss_epoch / len(train_dataloader),
            "train/pitch_loss_epoch": pitch_loss_epoch / len(train_dataloader),
            "train/velocity_loss_epoch": velocity_loss_epoch / len(train_dataloader),
            "train/dstart_loss_epoch": dstart_loss_epoch / len(train_dataloader),
            "train/duration_loss_epoch": duration_loss_epoch / len(train_dataloader),
        }

        model.eval()

        # val epoch
        val_metrics = validation_epoch(
            model,
            val_dataloader,
            masking_ratio,
            loss_lambdas,
            device,
        )
        val_metrics = {"val/" + key: value for key, value in val_metrics.items()}

        # maestro test epoch
        test_metrics = validation_epoch(
            model,
            maestro_test,
            masking_ratio,
            loss_lambdas,
            device,
        )
        test_metrics = {"maestro/" + key: value for key, value in test_metrics.items()}

        metrics = training_metrics | val_metrics | test_metrics
        wandb.log(metrics, step=step_count)

    # save model at the end of training
    save_checkpoint(model, optimizer, cfg, save_path=save_path)

    wandb.finish()

    # upload model to huggingface if specified in cfg
    if cfg.paths.hf_repo_id is not None:
        upload_to_huggingface(save_path, cfg)


if __name__ == "__main__":
    train()
