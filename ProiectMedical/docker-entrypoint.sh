#!/bin/bash
set -e

# Auto-build vectorstore if not present, so docker compose up works without manual intervention
if [ ! -d "/app/vectorstore" ] || [ -z "$(ls -A /app/vectorstore 2>/dev/null)" ]; then
    echo "Vectorstore not found. Building clinical index from corpus..."
    python scripts/build_index.py
    echo "Index built successfully."
else
    echo "Vectorstore already exists. Skipping rebuild."
fi

# Start the Streamlit application
exec streamlit run src/app.py --server.port=8501 --server.address=0.0.0.0
