import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain import PromptTemplate, HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA


from langchain_core.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline


FAISS_INDEX = "vectorstore/"

# Ensure FAISS index exists before loading
if not os.path.exists(FAISS_INDEX):
    raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX}. Run `ingest.py` first.")

# Prompt template
custom_prompt_template = """[INST] <<SYS>>
You are a trained bot to guide people about Indian Law. You will answer user's query with your knowledge and the context provided. 
If a question does not make any sense, or is not factually coherent, explain why instead of answering something incorrect. 
If you don't know the answer, do not share false information.
Do not say thank you or tell users that you are an AI Assistant.
<</SYS>>
Use the following pieces of context to answer the user's question:
Context: {context}
Question: {question}
Answer: [/INST]
"""

def set_custom_prompt_template():
    """
    Set the custom prompt template for the LLMChain
    """
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm():
    """
    Load the LLM
    """
    # repo_id = 'meta-llama/Llama-2-7b-chat-hf'
    repo_id = 'gpt2'  # Use a smaller model if needed

    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        device_map="auto",
        torch_dtype="auto"  # Automatically selects the best data type
    )

    tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=True)

    # Create pipeline
    pipe = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        # max_length=256,  # Reduce memory usage
        max_new_tokens=100, # Generate up to 100 new tokens
        temperature=0.3,  # More deterministic responses
        do_sample=False
    )

    return HuggingFacePipeline(pipeline=pipe)

def retrieval_qa_chain(llm, prompt, db):
    """
    Create the Retrieval QA chain
    """
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

def qa_pipeline():
    """
    Create the QA pipeline
    """
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.load_local(FAISS_INDEX, embeddings, allow_dangerous_deserialization=True)

    llm = load_llm()
    qa_prompt = set_custom_prompt_template()
    return retrieval_qa_chain(llm, qa_prompt, db)
