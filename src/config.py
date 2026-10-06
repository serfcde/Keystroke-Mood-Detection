"""Paths and constants shared by the whole pipeline."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

# Secrets such as ANTHROPIC_API_KEY live in ROOT/.env (git-ignored). A variable already set in the
# shell takes precedence over the file.
load_dotenv(ROOT / ".env")

# MOOD_RUN selects which dataset the pipeline uses; each run writes to its own output folders.
#   provided  (default) data/raw/synthetic_keystroke_raw_100k.csv, the course dataset
#   generated           data/raw/generated_keystroke_raw_100k.csv, from src/data_generator.py
RUN = os.environ.get("MOOD_RUN", "provided")
RAW_FILES = {
    "provided": ROOT / "data/raw/synthetic_keystroke_raw_100k.csv",
    "generated": ROOT / "data/raw/generated_keystroke_raw_100k.csv",
}
RAW_PATH = Path(os.environ.get("MOOD_RAW_PATH", RAW_FILES.get(RUN, RAW_FILES["provided"])))
_sub = "" if RUN == "provided" else RUN
CLEAN_PATH = ROOT / "data/processed" / _sub / "keystrokes_clean.csv"
WINDOW_PATH = ROOT / "data/processed" / _sub / "window_features.csv"
MODEL_DIR = ROOT / "models" / _sub
REPORT_DIR = ROOT / "reports" / _sub
FIG_DIR = REPORT_DIR / "figures"

# LLM explanation layer (src/llm.py)
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq").lower()   # groq | anthropic | template
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

WINDOW_S = 30            # PRD §9: one ML sample per 30-second window
PAUSE_MS = 500           # flight above this is a pause (EDA: typing gaps sit at 40-250 ms)
LONG_PAUSE_MS = 2000     # flight above this is a long pause
MAX_DWELL_MS = 1000      # a key held longer than this is treated as invalid
MIN_EVENTS_PER_WINDOW = 10
TREND_WINDOWS = 3        # previous windows used for temporal context (PRD §11)
RANDOM_STATE = 42

MOODS = ["calm", "positive", "stressed", "low"]

CORE_FEATURES = [
    "typing_speed", "mean_dwell", "std_dwell", "mean_flight", "std_flight",
    "mean_pause", "pause_count", "long_pause_count", "correction_rate",
]

# PRD §9 optional extensions. Digraph latency per character pair is not possible because characters
# are removed for privacy; char->char flight is the closest content-free equivalent.
EXTENDED_FEATURES = [
    "median_dwell", "dwell_cv", "flight_cv", "char_char_latency", "mean_burst_length", "key_entropy",
]
