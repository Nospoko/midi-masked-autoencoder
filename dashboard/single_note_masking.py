import numpy as np
import pandas as pd
import fortepyan as ff
import streamlit as st
from datasets import Dataset, load_dataset
from streamlit_pianoroll import from_fortepyan


def main():
    st.write("# Single note masking")
    dataset = get_dataset()

    record_idx = st.number_input(
        label="record id",
        min_value=0,
        max_value=len(dataset) - 1,
        value=0,
    )

    piece = ff.MidiPiece.from_huggingface(dataset[record_idx])

    note_idx = np.random.choice(piece.size)

    single_note_df = piece.df[note_idx : note_idx + 1].copy()
    single_note_piece = ff.MidiPiece(df=single_note_df)

    missing_note_df = piece.df.drop(note_idx)
    missing_note_piece = ff.MidiPiece(df=missing_note_df)

    cols = st.columns(2)
    with cols[0]:
        st.write("Unchanged input")
        from_fortepyan(
            piece=missing_note_piece,
            secondary_piece=single_note_piece,
            show_bird_view=False,
        )
        st.write("Original note")
        st.write(single_note_df.iloc[0].to_dict())

    target_pitch = single_note_df.iloc[0].pitch

    recreated_note = recreate_note_df(
        target_pitch=target_pitch,
        missing_note_df=missing_note_df,
    )
    recreated_note_df = pd.DataFrame([recreated_note])
    recreated_note_piece = ff.MidiPiece(df=recreated_note_df)

    with cols[1]:
        st.write("Algorithmic recreation")
        from_fortepyan(
            piece=missing_note_piece,
            secondary_piece=recreated_note_piece,
            show_bird_view=False,
        )
        st.write("Recreated note")
        st.write(recreated_note)
    st.write(piece.source)


def recreate_note_df(target_pitch: int, missing_note_df: pd.DataFrame) -> dict:
    # As random as possible, but make it audible
    velocity = np.random.choice(88) + 40
    start = np.random.random() * missing_note_df.end.max()
    max_duration = missing_note_df.end.max() - start
    duration = np.random.random() * max_duration
    end = start + duration

    note = {
        "pitch": target_pitch,
        "velocity": velocity,
        "start": start,
        "end": end,
    }
    return note


@st.cache_data
def get_dataset(split: str = "test") -> Dataset:
    dataset = load_dataset(
        path="pianoroll/sequenced-piano",
        name="basic-short",
        split=split,
    )

    return dataset


if __name__ == "__main__":
    main()
