import os
from langchain_groq import ChatGroq
from tenacity import retry, stop_after_attempt, wait_fixed
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key:
    print("Succeded!")

groq_llm = ChatGroq(
    model= "openai/gpt-oss-120b",
    temperature=0,
    api_key= groq_api_key,
    max_retries=3
)

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def invoke_structured(structured_llm, prompt):
    return structured_llm.invoke(prompt)