#!/bin/bash

# Start Ollama in the background.
/bin/ollama serve &

pid=$!

# Necessary pause for Ollama to start.
sleep 5

# Models to build into the image
# Pros: Reduces container spin up for new models
# Cons: Increases image size
model="llama2"
echo "Retrieving model $model..."
ollama pull "$model"
echo "$model retrieved."

# Wait for Ollama process to finish.
wait $pid
