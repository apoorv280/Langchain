from dotenv import load_dotenv
from typing import TypedDict
from langchain_groq import ChatGroq
# from langchain_classic.output_parsers import  StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableParallel

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

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)
prompt2 = PromptTemplate(
    template='Generate 5 short question and answers from the following text \n {text}',
    input_variables=['text']
)
prompt3 = PromptTemplate(
    template='merge the provided notes and quiz into a single document \n {notes} and {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()
parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz' : prompt2 | model2 | parser
})

merge_chain = prompt3 | model2 | parser

chain = parallel_chain | merge_chain

text = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

    Effective in high dimensional spaces.

    Still effective in cases where number of dimensions is greater than the number of samples.

    Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

    Versatile: different Kernel functions can be specified for the decision function. 
    Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

    If the number of features is much greater than the number of samples, 
    avoid over-fitting in choosing Kernel functions and regularization term is crucial.

    SVMs do not directly provide probability estimates, 
    these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support 
both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. 
However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, 
use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
"""

result = chain.invoke({'text':text})
print(result)

chain.get_graph().print_ascii()

