"""
Deprecated entry point - the Streamlit UI now lives in `app/main.py`.

Run with: streamlit run app/main.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import main  # noqa: E402

if __name__ == "__main__":
    main()
