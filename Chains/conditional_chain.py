from dotenv import load_dotenv
from typing import TypedDict, Literal
from langchain_groq import ChatGroq
# from langchain_classic.output_parsers import  StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import PydanticOutputParser,JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field

import os
import warnings
warnings.filterwarnings('ignore')



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
# from huggingface_hub import InferenceClient
# client = InferenceClient(api_key=token)
# models = client.list_deployed_models()
# print(models)

model1  = ChatGroq(
        temperature=0.01,
        model = "openai/gpt-oss-safeguard-20b",#"qwen/qwen3-32b",#"llama-3.3-70b-versatile",
        api_key=os.environ.get("GROQ_API_KEY")
    )

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b:cerebras",#"Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)
model2 = ChatHuggingFace(llm=llm, verbose=True)

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive','negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)




prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into positive and negative  \n {feedback} \n {format_instructions}',
    input_variables=['feedback'],
    partial_variables={'format_instructions':parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model1 | parser2



prompt2 = PromptTemplate(
    template='Write an appropirate response to this positive feedback  \n {feedback}',
    input_variables=['feedback']
    
)
prompt3 = PromptTemplate(
    template='Write an appropirate response to this negative feedback  \n {feedback}',
    input_variables=['feedback']
    
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model2 | parser),
    (lambda x:x.sentiment == 'negative', prompt3 | model2 | parser),
    RunnableLambda(lambda x: "Could not find any sentiment for the feedback")
)

chain = classifier_chain | branch_chain

result = (chain.invoke({'feedback':'This is a wonderfull smartphone'}))
print(result)

chain.get_graph().print_ascii()