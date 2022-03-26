import io
import logging
import queue
import threading

import matplotlib.pyplot as plt
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

from streamlit_webrtc import (
    RTCConfiguration,
    WebRtcMode,
    webrtc_streamer,
    AudioProcessorBase,
    VideoProcessorBase
)
import asyncio
from typing import List, Optional

logger = logging.getLogger(__name__)


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def main():
    st.write("""\
    # Streamlit Guitar Pedalboard
    
    Powered by [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc) & Spotify's [pedalboard](https://github.com/spotify/pedalboard)
    """)
    run_mode = st.radio('Trying modes', ('upload', 'sync', 'async'))
    if run_mode == 'sync':
        run_pedalboard()
    elif run_mode == 'async':
        run_pedalboard_async()
    elif run_mode == 'upload':
        upload_pedalboard()

    logger.debug("=== Alive threads ===")
    for thread in threading.enumerate():
        if thread.is_alive():
            logger.debug(f"  {thread.name} ({thread.ident})")

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



def run_pedalboard():
    """Transfer audio frames from the browser to the server and run them through pedalboard"""

    class AudioProcessor(AudioProcessorBase):
        board = None

        def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
            if self.board is not None:
                raw_samples = frame.to_ndarray()
                new_sample = self.board.process(raw_samples, sample_rate=frame.sample_rate, reset=False)
                new_sample = new_sample.astype(np.int16)
                new_frame = av.AudioFrame.from_ndarray(
                    new_sample, layout=frame.layout.name
                )
                new_frame.sample_rate = frame.sample_rate
            else:
                new_frame = frame
            return new_frame

    webrtc_ctx = webrtc_streamer(
        key="pedalboard-sync",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        audio_processor_factory=AudioProcessor,
        async_processing=True,
    )

    if webrtc_ctx.audio_processor:
        webrtc_ctx.audio_processor.board = make_board()


def run_pedalboard_async():

    class AudioProcessor(AudioProcessorBase):
        board = None

        async def process_frames(self, frames: List[av.AudioFrame]) -> List[av.AudioFrame]:
            raw_samples = np.concatenate((frame.to_ndarray() for frame in frames))
            new_samples = self.board.process(raw_samples, sample_rate=frames.sample_rate, reset=False)
            new_samples = new_samples.astype(np.int16)
            new_frames = [av.AudioFrame.from_ndarray(
                sample, layout=frames.layout.name, sample_rate = frames.sample_rate
            ) for sample in new_samples]
            return new_frames

        async def recv_queued(self, frames: List[av.AudioFrame]) -> List[av.AudioFrame]:
            if self.board is not None:
                logger.warning("Pre process")
                raw_samples = np.concatenate((frame.to_ndarray() for frame in frames))
                new_samples = self.board.process(raw_samples, sample_rate=frames.sample_rate, reset=False)
                new_samples = new_samples.astype(np.int16)
                new_frames = [av.AudioFrame.from_ndarray(
                    sample, layout=frames.layout.name, sample_rate = frames.sample_rate
                ) for sample in new_samples]
                logger.warning("post process")
            else:
                new_frames = frames
            return new_frames

    webrtc_ctx = webrtc_streamer(
        key="pedalboard-async",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        audio_processor_factory=AudioProcessor,
        async_processing=True,
    )

    if webrtc_ctx.audio_processor:
        logging.warning("Start Board Async")
        webrtc_ctx.audio_processor.board = make_board()

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

if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d:\n"
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()
