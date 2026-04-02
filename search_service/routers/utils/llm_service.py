from openai import AzureOpenAI
from dotenv import load_dotenv
import os


client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

response = client.chat.completions.create(
    model=os.getenv("AZURE_OPENAI_MODEL"),
    messages=[
        {"role": "user", "content": "Translate this sentence into Tamil in Latin Script: I love teaching AI."}
    ]
)
print(response.choices[0].message.content)