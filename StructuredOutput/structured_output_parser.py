# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict
from langchain_groq import ChatGroq
from langchain_classic.output_parsers import  StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate
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

schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic')
    
]
parser = StructuredOutputParser.from_response_schemas(schema)

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template="Give me 3 facts about the {topic} \n {format_instruction}",
    input_variables= ['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)


# prompt = template1.format()
# result = model.invoke(prompt)
# print(result)
# prompt1 = template1.invoke({'topic':'Black hole'})
# result1 = model.invoke(prompt1)
# prompt2 = template2.invoke({'text':result1.content})
# result2 = model.invoke(prompt2)


chain = template1 | llm | parser

result = chain.invoke({'topic':'black hole'})
print(result)