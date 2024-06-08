"""
A module to detect the speech in any language, and convert it to English
in predefined intervals.

Orginally: https://github.com/davabase/whisper_real_time/tree/master.
Some modifications are made to make it work with the latest version of
the libraries.
"""

from dataclasses import dataclass
from queue import Queue
from time import sleep
import threading
from sys import platform

import numpy as np
import speech_recognition as sr
import whisper
import torch

# Thread safe Queue for passing data from the threaded recording
# callback.
_MIC_QUEUE = Queue()

# Enable printing for main calls.
_PRINT_ENABLED: bool = True


def log(info: str) -> None:
    """Records the current step information,
    and logs it to console if desired."""
    if _PRINT_ENABLED:
        print(info)


@dataclass
class S2TSettings:
    """Settings for the speak to text functionality."""
    model: str = "base"
    language: str = "English"
    energy_threshold: int = 1000
    record_timeout: float = 20  # seconds
    linux_microphone: str = 'pulse'
    sample_rate: int = 16000  # Hz


@dataclass
class S2TResult:
    """Result of the speech to text functionality."""
    text: str
    data: bytes
    time_delta: float


def record_callback(_, audio: sr.AudioData) -> None:
    """
    Threaded callback function to receive audio data when
    recordings finish.
    audio: An AudioData containing the recorded bytes.
    """
    # Grab the raw bytes and push it into the thread safe queue.
    data = audio.get_raw_data()
    _MIC_QUEUE.put(data)


def speech2text(settings: S2TSettings,
                queue: Queue, mutex: threading.Lock) -> None:
    """
    This function is the main entry point for the speech to text
    functionality. It will record audio from the microphone and
    convert it to text in English. It should be run in a separate
    thread to prevent blocking the main thread. The function will
    use a queue to pass transcribed text back to the main thread.

    :param queue: A Queue to pass transcribed text back to the main thread.
    :param mutex: A Lock to stop while loop.
    :return: None
    """
    # We use SpeechRecognizer to record our audio because
    # it has a nice feature where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = settings.energy_threshold
    # Definitely do this, dynamic energy compensation lowers
    # the energy threshold dramatically to a point where the
    # SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False

    # Important for linux users.
    # Prevents permanent application hang and crash by using
    # the wrong Microphone
    if 'linux' in platform:
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            if settings.linux_microphone in name:
                source = sr.Microphone(
                    sample_rate=settings.sample_rate,
                    device_index=index)
                break
    else:
        source = sr.Microphone(sample_rate=settings.sample_rate)
    log("Microphone is acquired.")

    # Load / Download model
    model = settings.model
    if settings.model != "large" and \
            settings.language == "English":
        model = model + ".en"
    audio_model = whisper.load_model(model)
    log("Model download and load completed.")

    # Adjust for ambient noise.
    with source:
        recorder.adjust_for_ambient_noise(source, duration=3)
    log("Ambiant noise adjustment finished.")

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    recorder.listen_in_background(
        source,
        record_callback,
        phrase_time_limit=settings.record_timeout
    )

    # Initialize the transcription list.
    transcription = ['']

    # Wait for the mutex to be released.
    log("Listening... Press Ctrl+C to exit.")
    while mutex.locked():
        # Pull raw recorded audio from the queue.
        if not _MIC_QUEUE.empty():
            # Combine audio data from queue.
            audio_data = b''.join(_MIC_QUEUE.queue)
            _MIC_QUEUE.queue.clear()
            log(f"Audio data is {"empty" if not audio_data else "not empty"}.")

            # Find the duration of the audio_data.
            audio_duration: float = len(
                audio_data) / 2 / float(settings.sample_rate)
            log(f"Duration of audio data: {audio_duration} seconds.")

            # Convert in-ram buffer to something the model can use
            # directly without needing a temp file. Convert data from
            # 16 bit wide integers to floating point with a width of
            # 32 bits. Clamp the audio stream frequency to a PCM
            # wavelength compatible default of 32768 Hz max.
            # https://github.com/openai/whisper/blob/ba3f3cd54b0e5b8ce1ab3de13e32122d0d5f98ab/whisper/audio.py#L62
            audio_np = np\
                .frombuffer(audio_data, dtype=np.int16)\
                .astype(np.float32) / 32768.0

            # Read the transcription.
            result = audio_model.transcribe(
                audio_np,
                fp16=torch.cuda.is_available()
            )
            text = result['text'].strip()
            log(f"Fetched text: {text}")

            # If we detected a pause between recordings, add a new
            # item to our transcription. Otherwise edit the existing one.
            transcription.append(text)
            queue.put(S2TResult(
                text=text,
                data=audio_data,
                time_delta=audio_duration,
            ))
            log(f"Queue size: {queue.qsize()}")
        else:
            # Infinite loops are bad for processors, must sleep.
            sleep(0.25)


if __name__ == "__main__":
    # Enable log printing since it is executable run.
    _PRINT_ENABLED = True

    # Create a locked mutex to control the while loop in the
    # process function.
    mutex = threading.Lock()
    mutex.acquire()

    # Create a queue to pass transcribed text back to the main thread.
    main_queue = Queue()

    # Start the process function in a separate thread.
    thread = threading.Thread(
        target=speech2text,
        args=(S2TSettings(), main_queue, mutex)
    )
    thread.start()

    while True:
        try:
            print(main_queue.get())
        except KeyboardInterrupt:
            mutex.release()
            print("Exiting...")
            break

    #  Wait for the thread to finish.
    thread.join()
