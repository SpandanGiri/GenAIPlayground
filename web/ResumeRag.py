from langchain_ollama import OllamaEmbeddings
from uuid import uuid4
from numpy.linalg import norm
import numpy as np

from langchain_core.documents import Document
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
import Cleanup as cl


filenames = [
    'Spandan_Giri_Resume.pdf',
    'Saptadeep_Banerjee_resume.pdf',
    'Girish_Garg_Resume.pdf.pdf',
    '2019145_Shivam_Sourav_Jha.pdf'
]

documents =[]

for file in filenames:
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    for page in pages:
        documents.append(page)

def clean_document(text):
    
    text = cl.fix_line_breaks(text)
    text = cl.clean_bullets_and_spaces(text)
    #text = cl.extract_relevant_sections(text)
    return text.strip()

for i, doc in enumerate(documents):

    documents[i] = Document(page_content=clean_document(doc.page_content), metadata=doc.metadata)

print(clean_document(documents[2].page_content))

#vector db

embeddings = OllamaEmbeddings(model="llama3.2")

d = len(embeddings.embed_query("test query"))
index = faiss.IndexFlatL2(d)

query = "where did Spandan studied from?"

query_embeddings = np.array(embeddings.embed_query(query))
embedding1 = np.array(embeddings.embed_query(documents[0].page_content))
embedding2 = np.array(embeddings.embed_query(documents[1].page_content))

distance1q = norm(query_embeddings-embedding1)
distance2q = norm(query_embeddings-embedding2)


print(f"norm dist embd1 embdq {distance1q}")
print(f"norm dist embd2 embdq {distance2q}")

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

uuids = [str(uuid4()) for _ in range(len(documents))]
vector_store.add_documents(documents=documents, ids=uuids)



#operations

results = vector_store.similarity_search(
    query,
    k=3,
)

context = results[0].page_content
#print("context doc: "+ context)


template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n{context}\n\nQuestion: {question}\nHelpful Answer:"""
prompt = ChatPromptTemplate.from_template(template)
model = OllamaLLM(model="llama3.2")
chain = prompt | model 


print("ans "+chain.invoke({"question":query,"context": context}))