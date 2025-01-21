from langchain_core.documents import Document
from langchain.embeddings import HuggingFaceEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate , PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain.chains import LLMChain, StuffDocumentsChain , RetrievalQA
from langchain.chains.combine_documents import  create_stuff_documents_chain


# #embedding
embeddings = HuggingFaceEmbeddings()


try:
    vector_store = FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
except:
    filename = "interviewBit_DataScience.pdf"
    loader = PyPDFLoader(filename)
    pages = loader.load()
    text_spliter = SemanticChunker(HuggingFaceEmbeddings())

    chunks = text_spliter.split_documents(pages)

    print(len(chunks))

    d = len(embeddings.embed_query("test query"))
    index = faiss.IndexFlatL2(d)


    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    vector_store.add_documents(chunks)
    vector_store.save_local("faiss_index")


# #query
query = "what is overfitting and underfitting ?"

results = vector_store.similarity_search(
    query,
    k=3,
)

print(results)

# #ollama

llm = model = OllamaLLM(model="llama3.2")

template = """
1. Use the following pieces of context to answer the question at the end.
2. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.\n
3. Keep the answer crisp and limited to 3,4 sentences.

Context: {context}

Question: {question}

Helpful Answer:"""

prompt = ChatPromptTemplate.from_template(template)

doc_chain = create_stuff_documents_chain(llm,prompt)

query = "What is underfitting and over fitting?"

print(doc_chain.invoke({"context":results,"question":query}))
