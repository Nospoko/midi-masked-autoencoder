import os
import argparse

import torch
import numpy as np
import pandas as pd
import fortepyan as ff
import streamlit as st
from datasets import load_dataset
from torch.utils.data import Subset, DataLoader

from data.dataset import MidiDataset
from models.mae import MidiMaskedAutoencoder


def makedir_if_not_exists(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)


def preprocess_dataset(
    dataset_name: str,
    query: str,
):
    dataset = load_dataset(dataset_name, split="validation")

    ds = MidiDataset(dataset)

    idx_query = [i for i, name in enumerate(ds.dataset["source"]) if str.lower(query) in str.lower(name)]

    ds = Subset(ds, indices=list(idx_query))

    return ds


def display_audio(title, midi_files: list[str], mp3_files: list[str]):
    st.title(title)

    cols = st.columns([2, 2])
    fig_titles = ["### Original", "### Model"]

    for i, col in enumerate(cols):
        with col:
            st.write(fig_titles[i])
            piece = ff.MidiFile(midi_files[i]).piece
            fig = ff.view.draw_pianoroll_with_velocities(piece)
            st.pyplot(fig)
            st.audio(mp3_files[i], format="audio/mp3")


def denormalize_velocity(velocity: np.ndarray):
    return ((velocity + 1) * 64).astype("int")


def to_midi_piece(pitch: np.ndarray, dstart: np.ndarray, duration: np.ndarray, velocity: np.ndarray) -> ff.MidiPiece:
    record = {
        "pitch": pitch,
        "velocity": velocity,
        "dstart": dstart,
        "duration": duration,
    }

    df = pd.DataFrame(record)
    df["start"] = df.dstart.cumsum().shift(1).fillna(0)
    df["end"] = df.start + df.duration

    return ff.MidiPiece(df)


def process_files_based_on_query(
    model: MidiMaskedAutoencoder,
    dataset_name: str,
    query: str,
    save_dir: str,
    device: torch.device,
):
    dataset = preprocess_dataset(
        dataset_name,
        query=query,
    )

    dataloader = DataLoader(dataset, batch_size=1024, num_workers=8, shuffle=False)

    with torch.no_grad():
        for batch in dataloader:
            pitches = batch["pitch"].to(device)
            velocities = batch["velocity"].to(device)
            dstarts = batch["dstart"].to(device)
            durations = batch["duration"].to(device)

            pred_pitches, pred_velocities, pred_dstarts, pred_durations, mask = model(
                pitch=pitches,
                velocity=velocities,
                dstart=dstarts,
                duration=durations,
                masking_ratio=0.5,
            )
            mask = mask.detach().bool()
            pred_pitches = torch.argmax(pred_pitches, dim=-1)

            # gen_pitches = pred_pitches
            # gen_velocities = pred_velocities
            # gen_dstarts = pred_dstarts
            # gen_durations = pred_durations

            # replace tokens that were masked with generated values
            gen_pitches = torch.where(mask, pred_pitches, pitches)
            gen_velocities = torch.where(mask, pred_velocities, velocities)
            gen_dstarts = torch.where(mask, pred_dstarts, dstarts)
            gen_durations = torch.where(mask, pred_durations, durations)

            for i in range(len(pitches)):
                pitch = pitches[i].cpu().numpy() + 21
                velocity = velocities[i].cpu().numpy()
                dstart = dstarts[i].cpu().numpy()
                duration = durations[i].cpu().numpy()
                gen_pitch = gen_pitches[i].cpu().numpy() + 21
                gen_velocity = gen_velocities[i].cpu().numpy()
                gen_dstart = gen_dstarts[i].cpu().numpy()
                gen_duration = gen_durations[i].cpu().numpy()

                velocity = denormalize_velocity(velocity)
                gen_velocity = denormalize_velocity(gen_velocity)

                original_piece = to_midi_piece(pitch, dstart, duration, velocity)
                model_piece = to_midi_piece(gen_pitch, gen_dstart, gen_duration, gen_velocity)

                original_midi = original_piece.to_midi()
                model_midi = model_piece.to_midi()

                # save as midi
                original_midi.write(f"{save_dir}/original/{query}-{i}-original.midi")
                model_midi.write(f"{save_dir}/generated/{query}-{i}-model.midi")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str)
    args = parser.parse_args()

    checkpoint = torch.load("checkpoints/mae10m-2023-11-08-20-29-params-9.88M.ckpt")

    cfg = checkpoint["config"]
    # device = cfg.train.device
    device = "cpu"

    model = MidiMaskedAutoencoder(
        encoder_dim=cfg.model.encoder_dim,
        encoder_depth=cfg.model.encoder_depth,
        encoder_num_heads=cfg.model.encoder_num_heads,
        decoder_dim=cfg.model.decoder_dim,
        decoder_depth=cfg.model.decoder_depth,
        decoder_num_heads=cfg.model.decoder_num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
    ).to(device)

    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset_name = "JasiekKaczmarczyk/maestro-v1-sustain-masked"
    save_dir = "tmp/midi"

    makedir_if_not_exists(f"{save_dir}/generated")
    makedir_if_not_exists(f"{save_dir}/original")

    process_files_based_on_query(model, dataset_name=dataset_name, query=args.query, save_dir="tmp/midi", device=device)


if __name__ == "__main__":
    main()