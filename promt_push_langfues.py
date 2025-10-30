from langfuse import Langfuse
import os
from langfuse.model import ChatMessageWithPlaceholdersDict
from dotenv import load_dotenv

# ✅ Load environment variables from .env file
load_dotenv()
# LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST")
)


# Define the prompt
prompt_messages: list[ChatMessageWithPlaceholdersDict] = [
    {
        "role": "system",
        "content": (
            "You are a log analysis expert. Combine the following two log analysis summaries "
            "into one comprehensive summary. Identify common patterns, prioritize the most "
            "critical issues, and provide actionable insights."
        ),
    },
    {
        "role": "user",
        "content": (
            "Please combine these two log analysis summaries into one comprehensive summary:\n\n"
            "PART 1 ANALYSIS:\n{{summary_part1}}\n\n"
            "PART 2 ANALYSIS:\n{{summary_part2}}\n\n"
            "Provide a unified summary that:\n"
            "1. Highlights the most critical issues across both parts\n"
            "2. Identifies patterns or connections between the parts\n"
            "3. Prioritizes issues by severity\n"
            "4. Provides actionable recommendations"
        ),
    },
]


# ✅ Create or update your prompt
prompt = langfuse.create_prompt(
    name="combine_log_analysis_summaries",  # Prompt unique name
    labels=["production"],                     # Version label (e.g., production, dev, test)
    type="chat",                            # or "completion" depending on your model
    config={
        "model": "gpt-4o-mini",             # Default model for this prompt (optional)
        "temperature": 0.7
    },
    prompt=prompt_messages,
)

print("✅ Prompt successfully pushed to Langfuse!")
print(f"Prompt name: {prompt.name}")
print(f"Prompt labels: {prompt.labels}")
print(f"Prompt version ID: {prompt.version}")
