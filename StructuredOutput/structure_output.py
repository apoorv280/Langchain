from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict
# from ChatModels.chatmodel_huggingface_api import LLM

load_dotenv()
LLM = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation"
)

model1  = ChatHuggingFace(llm = LLM)

#schema

class Review(TypedDict):
    summary:str
    sentiment:str

structured_model = model1.with_structured_output(Review)

result = structured_model.invoke("The ceiling fan is great but it makes a lot of noise when it is on high speed. The customer service is also very slow and takes a lot of time to respond. Overall, I am not satisfied with the product and the service.")

print(result)
print(result.summary)
print(result.sentiment)


