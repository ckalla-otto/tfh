"""Single-image face Presentation Attack Detection (PAD) package.

Importing `pad` loads the local `.env` (via python-dotenv) so the Kaggle
credential env vars are available to every CLI without manual sourcing.
"""
from .utils import load_env

load_env()