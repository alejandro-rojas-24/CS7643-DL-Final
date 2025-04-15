#!/bin/bash

# Start Ollama in the background.
/bin/ollama serve &

pid=$!

# Pause for Ollama to start.
sleep 5

model="llama2"
echo "Retrieving model $model..."
ollama pull "$model"
echo "$model retrieved."

# Wait for Ollama process to finish.
wait $pid
