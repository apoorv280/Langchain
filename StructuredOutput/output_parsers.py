from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from huggingface_hub import InferenceClient
import os

# from ChatModels.chatmodel_huggingface_api import LLM

load_dotenv()
# client = InferenceClient(
#     model= "Qwen/Qwen2.5-7B-Instruct"
#     )
# print(client)

llm  = ChatGroq(
        temperature=0.01,
        model = "openai/gpt-oss-safeguard-20b",
        api_key=os.environ.get("GROQ_API_KEY")
    )

# LLM = HuggingFaceEndpoint(
#     repo_id="Qwen/Qwen2.5-7B-Instruct",
#     task="text-generation"
# )

# model = ChatHuggingFace(llm = llm)

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template="write a detailed report on {topic}",
    input_variables= ['topic']
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template="write a 5 line summary on the following {text}",
    input_variables= ['text']
)

# prompt1 = template1.invoke({'topic':'Black hole'})
# result1 = model.invoke(prompt1)
# prompt2 = template2.invoke({'text':result1.content})
# result2 = model.invoke(prompt2)

parser = StrOutputParser()
chain = template1 | llm | parser | template2 | llm | parser

result = chain.invoke({'topic':'Black hole'})
print(result)
