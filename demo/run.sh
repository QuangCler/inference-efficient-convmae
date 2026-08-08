#!/bin/bash
# Launch the demo (Linux/macOS). Assumes requirements installed + checkpoints fetched.
cd "$(dirname "$0")"
[ -d venv ] && source venv/bin/activate
python app.py
