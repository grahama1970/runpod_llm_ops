curl -X POST https://jqy6nr0djajtlf-8000.proxy.runpod.net/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
  "model": "Qwen/Qwen2.5-0.5B",
  "messages": [
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "max_tokens": 50,
  "temperature": 0.7
}'