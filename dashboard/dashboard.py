import json
import pickle
from glob import glob

import torch
import pandas as pd
import fortepyan as ff
import streamlit as st
from omegaconf import OmegaConf
from datasets import Dataset, load_dataset
from streamlit_pianoroll import from_fortepyan

from data.dataset import MidiDataset
from models.mae import MidiMaskedAutoencoder


def display_pianoroll(title: str, midi_pieces: dict[ff.MidiPiece, ff.MidiPiece]):
    st.title(title)

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Original")
        from_fortepyan(midi_pieces["original"])
        fig = ff.view.draw_dual_pianoroll(midi_pieces["original"])
        st.pyplot(fig)

    with col2:
        st.write("### Generated")
        from_fortepyan(midi_pieces["generated"])
        fig = ff.view.draw_dual_pianoroll(midi_pieces["generated"])
        st.pyplot(fig)


def not_main():
    st.set_page_config(layout="wide")

    with open("tmp/processed_files.pickle", "rb") as handle:
        unserialized_data = pickle.load(handle)

    files = unserialized_data.keys()

    selected_filename = st.selectbox("Select piece to display", options=files)

    display_pianoroll(
        title=selected_filename,
        midi_pieces=unserialized_data[selected_filename],
    )


@st.cache_data
def get_model(checkpoint_path: str) -> MidiMaskedAutoencoder:
    checkpoint = torch.load(checkpoint_path)

    cfg = checkpoint["config"]
    st.json(OmegaConf.to_container(cfg), expanded=False)

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

    return model


def model_selection() -> MidiMaskedAutoencoder:
    checkpoints = glob("checkpoints/*")
    checkpoint_path = st.selectbox("Select checkpoint", options=checkpoints)

    model = get_model(checkpoint_path)
    return model


@st.cache_data
def get_dataset() -> Dataset:
    dataset_name = "JasiekKaczmarczyk/maestro-v1-sustain-masked"
    dataset = load_dataset(dataset_name, split="validation")
    return dataset


def main():
    # Load model
    _ = model_selection()

    # Prep data
    dataset = get_dataset()
    source = [json.loads(source) for source in dataset["source"]]
    source_df = pd.DataFrame(source)

    composers = source_df.composer.unique()
    selected_composer = st.selectbox(
        label="Select composer",
        options=composers,
        index=3,
    )

    ids = source_df.composer == selected_composer
    piece_titles = source_df[ids].title.unique()
    selected_title = st.selectbox(
        label="Select title",
        options=piece_titles,
    )
    st.write(selected_title)

    ids = (source_df.composer == selected_composer) & (source_df.title == selected_title)
    n_samples = 10
    seed = 137
    part_df = source_df[ids].sample(n_samples, random_state=seed)

    idxs = part_df.index.values
    part_dataset = dataset.select(idxs)
    _ = MidiDataset(part_dataset)


if __name__ == "__main__":
    main()
