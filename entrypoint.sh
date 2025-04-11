#!/bin/bash

# Start Ollama in the background.
/bin/ollama serve &

pid=$!

# Pause for Ollama to start.
sleep 5

# Pull down model
echo "Retrieving model..."
ollama pull llama3.2:1b
echo "Model retrieved."

# Wait for Ollama process to finish.
wait $pid
