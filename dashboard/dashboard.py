import pickle

import fortepyan as ff
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_pianoroll import from_fortepyan


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


def main():
    st.set_page_config(layout="wide")

    with open("tmp/processed_files.pickle", "rb") as handle:
        unserialized_data = pickle.load(handle)

    files = unserialized_data.keys()

    selected_filename = st.selectbox("Select piece to display", options=files)

    display_pianoroll(
        title=selected_filename,
        midi_pieces=unserialized_data[selected_filename],
    )


if __name__ == "__main__":
    main()
