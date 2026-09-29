# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict
from langchain_groq import ChatGroq
# from langchain_classic.output_parsers import  StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
# from langchain_core.output_parsers import JsonOutputParser
# from huggingface_hub import InferenceClient
# from huggingface_hub import HfApi
import os

# from ChatModels.chatmodel_huggingface_api import LLM

load_dotenv()



# # 1. Manually pull whatever variable your token is stored under
# token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")

# # 2. Hard-inject it into the os environment state so the chat router sees it
# os.environ["HF_TOKEN"] = token

#print(os.getenv("HUGGINGFACEHUB_API_TOKEN"))
# client = InferenceClient(
#     model= "Qwen/Qwen2.5-7B-Instruct"
#     )
# print(client)

llm  = ChatGroq(
        temperature=0.01,
        model = "openai/gpt-oss-safeguard-20b",#"qwen/qwen3-32b",#"llama-3.3-70b-versatile",
        api_key=os.environ.get("GROQ_API_KEY")
    )


class Person(BaseModel):

    name : str = Field(description='NAme of the person')
    age: int = Field(gt=18, description='Age of the person')
    city: str = Field(description='Name of the city the person belong to')

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate the name, age and city of a fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

# prompt = template.invoke({'place': 'indian'})

# print(prompt)

chain = template | llm | parser

result = chain.invoke({'place':'indian'})

result = print(result)
