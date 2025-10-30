import os
import uuid
from openai import OpenAI

# Generate a trace ID
trace_id = str(uuid.uuid4())

# Use your env vars or replace directly
BASE_URL = "https://staging.worker.tailwinds.ai/llm/ModelProfile1"
API_KEY = "md-c89c62b9-0ada-42ba-b0d0-f6dbd0167393"

# Initialize client
client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,  # still required for SDK initialization
)

# --- Chat request (✅ includes correct Authorization header) ---
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a log analysis assistant."},
        {"role": "user", "content": "Summarize the following logs: example error logs..."},
    ],
    temperature=1.0,
    max_tokens=512,
    extra_headers={
        "X-Trace-ID": trace_id}
)

print("\n✅ Model Response:")
print(response.choices[0].message.content)
