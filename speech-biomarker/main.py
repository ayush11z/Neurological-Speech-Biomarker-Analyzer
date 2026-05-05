import os
from dotenv import load_dotenv
from deepgram import DeepgramClient, PrerecordedOptions

load_dotenv()

_client = None


def _get_client() -> DeepgramClient:
    global _client
    if _client is None:
        _client = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))
    return _client


def transcribe(audio_path: str, model: str = "nova-3"):
    """Return the full Deepgram response object for a WAV file."""
    with open(audio_path, "rb") as f:
        buf = f.read()

    options = PrerecordedOptions(model=model, smart_format=True, language="en")
    return _get_client().listen.rest.v("1").transcribe_file({"buffer": buf}, options)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "sample.wav"
    resp = transcribe(path)
    print(resp.results.channels[0].alternatives[0].transcript)