from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 1. Load PDF
loader = PyPDFLoader("assignment.pdf.pdf")
pages = loader.load_and_split()

# 2. Create Embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 3. Vector DB
db = FAISS.from_documents(pages, embeddings)

# 4. Load LLM
llm = Ollama(model="tinyllama")

# 5. Create custom prompt
prompt_template = """
Use the following pieces of context to answer the question at the end. 
Include bullet points in the answer, and cite sources using [source number].

If you don’t know the answer, just say you don’t know. Don’t try to make up an answer.

{context}

Question: {question}

Helpful Answer with bullet points, disclaimer, and sources:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# 6. Setup RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

# 7. Ask Questions
while True:
    query = input("\n🔷 Enter your question (or type 'exit' to quit): ")
    if query.lower() == "exit":
        break
    answer = qa_chain.run(query)
    print("\n📌 Answer:\n", answer)
