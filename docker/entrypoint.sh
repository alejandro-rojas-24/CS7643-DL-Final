#!/bin/bash

# Start Ollama in the background.
/bin/ollama serve &

pid=$!

# Necessary pause for Ollama to start.
sleep 5

# Models to build into the image
# Pros: Reduces container spin up for new models
# Cons: Increases image size

model="gemma3:1b"
echo "Retrieving model $model..."
ollama pull "$model"
echo "$model retrieved."

model="gemma3:4b"
echo "Retrieving model $model..."
ollama pull "$model"
echo "$model retrieved."

# model="gemma3:4b"
# echo "Retrieving model $model..."
# ollama pull "$model"
# echo "$model retrieved."

# Wait for Ollama process to finish.
wait $pid
