from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from huggingface_hub import InferenceClient
from huggingface_hub import HfApi
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

# llm = HuggingFaceEndpoint(
#     repo_id="Qwen/Qwen2.5-7B-Instruct",
#     task="text-generation",
#     # serverless_provider="hf-inference",
#     huggingfacehub_api_token=token,
#     temperature=0.01,
#     max_new_tokens=512
# )


# model = ChatHuggingFace(llm = llm, api_key=token)
parser = JsonOutputParser()

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template="Give me the name, age, city of a fictional person \n {format_instruction}",
    input_variables= [],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# # 2nd prompt -> summary
# template2 = PromptTemplate(
#     template="write a 5 line summary on the following {text}",
#     input_variables= ['text']
# )

prompt = template1.format()
# result = model.invoke(prompt)
# print(result)
# prompt1 = template1.invoke({'topic':'Black hole'})
# result1 = model.invoke(prompt1)
# prompt2 = template2.invoke({'text':result1.content})
# result2 = model.invoke(prompt2)


chain = template1 | llm | parser

result = chain.invoke({})
print(result)