"""Command-line entry point for the traceable website replay subset."""
from pathlib import Path

from src.data_loader import build_app_data


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(build_app_data(root))

