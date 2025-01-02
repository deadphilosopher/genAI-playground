#!/bin/bash

curl http://localhost:11434/api/chat -d '{
  "model": "llama3.2",
  "messages": [
    { "role": "user", "content": "What is Big O complexity?" }
  ]
}'
