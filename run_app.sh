#!/usr/bin/env bash
# ReviewSense AI - One-Click Application Launcher

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "=================================================="
echo " 🎬 Launching ReviewSense AI"
echo "=================================================="

# Check if virtualenv exists; if not, create one
if [ ! -d "venv" ]; then
    echo "Creating virtual environment 'venv'..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements if streamlit is missing
if ! python -c "import streamlit" 2>/dev/null; then
    echo "Installing project dependencies..."
    pip install -r requirements.txt
fi

echo "Starting Streamlit web interface..."
streamlit run app.py
