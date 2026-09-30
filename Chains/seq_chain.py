from dotenv import load_dotenv
from typing import TypedDict
from langchain_groq import ChatGroq
# from langchain_classic.output_parsers import  StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
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


prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

parser = StrOutputParser()
chain = prompt1 | llm | parser | prompt2 | llm | parser

result = chain.invoke({'topic': 'Unemployment in India'})
print(result)
chain.get_graph().print_ascii()