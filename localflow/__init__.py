"""LocalFlow - a fully local, open-source voice dictation app.

Hold a hotkey, speak, release: your words appear - formatted - in whatever
app has focus. Powered by OpenAI Whisper running entirely on your machine.
"""

__version__ = "0.1.0"

from .app import DictationEvent, FlowController
from .config import Config

__all__ = ["FlowController", "DictationEvent", "Config", "__version__"]
