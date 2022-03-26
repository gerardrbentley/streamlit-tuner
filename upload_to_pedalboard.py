import io

import numpy as np
import streamlit as st
import av
from pedalboard import (
    Pedalboard,
    Chorus,
    Compressor,
    Convolution,
    Distortion,
    Gain,
    HighpassFilter,
    LadderFilter,
    Limiter,
    LowpassFilter,
    NoiseGate,
    Phaser,
    PitchShift,
    Reverb
)


def make_board():
    st.checkbox('Compressor', True, key='compressor')
    st.checkbox('Gain', True, key='gain')
    st.checkbox('Chorus', True, key='chorus')
    st.checkbox('LadderFilter', True, key='ladderfilter')
    st.checkbox('Phaser', True, key='phaser')
    st.checkbox('Reverb', True, key='reverb')
    # st.checkbox('PitchShift', True, key='pitchshift')
    board = Pedalboard([])
    if st.session_state.compressor:
        board.append(Compressor(threshold_db=-40, ratio=25))
    if st.session_state.gain:
        board.append(Gain(gain_db=40))
    if st.session_state.chorus:
        board.append(Chorus())
    if st.session_state.ladderfilter:
        board.append(LadderFilter(mode=LadderFilter.Mode.HPF12, cutoff_hz=900))
    if st.session_state.phaser:
        board.append(Phaser())
    if st.session_state.reverb:
        board.append(Reverb(room_size=0.6))
    # if st.session_state.pitchshift:
    #     board.append(PitchShift(semitones=2))

    # st.write('pedalboard', board)
    return board


def upload_pedalboard():
    """Don't play sound live, just upload file :("""
    file = st.file_uploader('Upload sound file', ['.mp3', '.wav', '.m4a'], accept_multiple_files=False)
    if file is not None:
        raw_audio = file.read()
        st.subheader('Raw Input Audio')
        st.audio(raw_audio)
        container = av.open(io.BytesIO(raw_audio))
        board = make_board()
        raw_samples = []
        processed_samples = []
        output_frames = []
        for frame in container.decode(audio=0):
            sample = frame.to_ndarray()
            processed_sample = board.process(sample, sample_rate=frame.sample_rate, reset=False)
            out_frame = av.AudioFrame.from_ndarray(
                processed_sample, format=frame.format.name, layout=frame.layout.name
            )
            out_frame.sample_rate = frame.sample_rate
            raw_samples.append(sample)
            processed_samples.append(processed_sample)
            output_frames.append(out_frame)


        output_container = av.open('processed.mp3', 'wb')
        output_stream = output_container.add_stream('mp3')
        for frame in output_frames:
            for packet in output_stream.encode(frame):
                output_container.mux(packet)
        for packet in output_stream.encode(None):
            output_container.mux(packet)
        output_container.close()

        processed_bytes = open('processed.mp3', 'rb').read()
        st.audio(processed_bytes)

        st.download_button('Download processed audio', processed_bytes)
        st.line_chart(np.concatenate([x.squeeze() for x in raw_samples]))
        st.line_chart(np.concatenate([x.squeeze() for x in processed_samples]))

st.write("""\
# Streamlit Guitar Pedalboard

Powered by Spotify's [pedalboard](https://github.com/spotify/pedalboard) library
""")
upload_pedalboard()