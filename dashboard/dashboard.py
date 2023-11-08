import glob

import fortepyan as ff
import streamlit as st


def display_pianoroll(title, orginal_midi: str, generated_midi: str):
    st.title(title)

    col1, col2 = st.columns([2, 2])

    with col1:
        st.write("### Original")
        piece = ff.MidiFile(orginal_midi).piece
        fig = ff.view.draw_pianoroll_with_velocities(piece)
        st.pyplot(fig)

    with col2:
        st.write("### Model")
        piece = ff.MidiFile(generated_midi).piece
        fig = ff.view.draw_pianoroll_with_velocities(piece)
        st.pyplot(fig)


def main():
    midi_files = glob.glob("tmp/midi/original/**")

    # get only filenames without extension, model type and dir
    filenames = [name.replace(".midi", "").replace("-original", "").split("/")[-1] for name in midi_files if "original" in name]

    selected_filename = st.selectbox("Select piece to display", options=filenames)

    original_midi = f"tmp/midi/original/{selected_filename}-original.midi"
    genrated_midi = f"tmp/midi/generated/{selected_filename}-model.midi"

    display_pianoroll(
        title=selected_filename,
        orginal_midi=original_midi,
        generated_midi=genrated_midi,
    )


if __name__ == "__main__":
    main()
