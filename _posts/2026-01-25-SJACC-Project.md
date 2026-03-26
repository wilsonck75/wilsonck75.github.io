---
layout: post
title: "Advanced Arabic RAG Pipeline"
date: 2026-01-25
image: "posts/arabic-rag-theme.png"
permalink: /projects/advanced-arabic-rag-pipeline/
categories: [python, rag, ai, nlp]
tags: [RAG, Arabic NLP, AI, Python, Vector Search, ChromaDB]
---

# Advanced Arabic RAG Pipeline: A Technical Walkthrough

In the rapidly evolving landscape of legal tech and data analysis, bridging the gap between massive document archives and actionable insights is a critical challenge. This is especially true when dealing with multilingual datasets—specifically those mixing English and Arabic—where traditional keyword search often falls short.

This project demonstrates the construction of an **Advanced Retrieval-Augmented Generation (RAG) Pipeline** designed to ingest, process, and intelligent query complex institutional data.

## The Challenge

We faced a common problem: huge volumes of unstructured data (PDFs, text files) and structured export data that needed to be searchable in natural language. The system needed to:
1.  **Handle Multilingual Content**: Seamlessly process and retrieve both Arabic and English text.
2.  **Maintain Context**: Understand the semantic meaning of queries, not just match keywords.
3.  **Provide Bilingual Answers**: Generate responses in both English and Arabic to support a diverse team.

## Architecture Guidelines

Our solution leverages a modern Python stack centered around **LangChain** for orchestration and **ChromaDB** for vector storage.

1.  **Ingestion**: Load text, PDFs, and OCR'd images.
2.  **Processing**: Split documents into semantic chunks.
3.  **Embedding**: Convert text to vectors using a multilingual model (`intfloat/multilingual-e5-small`).
4.  **Storage**: Index vectors in ChromaDB.
5.  **Retrieval & Generation**: Use an LLM (GPT-4o) to synthesize answers from retrieved context.

---

## Step 1: Data Ingestion & Chunking 📥

The foundation of any RAG system is how it handles data. We built a robust ingestion pipeline capable of reading multiple file formats.

We use `RecursiveCharacterTextSplitter` to break documents into manageable chunks. This is crucial—chunks that are too large confuse the model, while chunks that are too small lose context.

```python
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)

# Example Loading Logic
documents = []
for txt_path in glob.glob(os.path.join(DATA_DIR, "*.txt")):
    loader = TextLoader(txt_path, encoding="utf-8")
    documents.extend(loader.load())
```

## Step 2: Building the Vector Index 🧠

Once we have our chunks, we need to convert them into mathematical vectors. For this project, we chose the **multilingual-e5-small** model. It's lightweight but highly effective for semantic similarity across languages.

We use **ChromaDB** as our local vector store. It handles the complexity of indexing and nearest-neighbor search.

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
DB_DIR = "backend/chroma_db"

def ingest():
    # ... loading and splitting code ...
    
    print("Initializing embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Creating vector store...")
    vector_store = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name="arabic_rag"
    )
    
    # Add documents to the store (automatically computes embeddings)
    vector_store.add_documents(documents=chunks)
    print(f"Ingestion complete. Vector store saved to {DB_DIR}")
```

## Step 3: The RAG Retrieval Chain 🔗

With our data indexed, we can build the retrieval chain. This is where the magic happens. We retrieve the most relevant 5 chunks (`k=5`) and pass them as context to the LLM.

We enforce a strict Bilingual Output format using a custom system prompt.

```python
class ArabicRAG:
    def __init__(self, llm=None):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vector_store = Chroma(
            persist_directory=DB_DIR,
            embedding_function=self.embeddings,
            collection_name="arabic_rag"
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        
        # We use GPT-4o for its superior multilingual reasoning
        self.llm = llm if llm else ChatOpenAI(model="gpt-4o", temperature=0)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intelligent knowledge assistant. 
            
            STRICT OUTPUT FORMAT:
            You must provide every answer in TWO languages:
            1. English Answer
            2. Arabic Answer (Translation)
            
            Use the following format exactly:
            **English Answer:**
            [Your answer here]
            
            **الإجابة بالعربية:**
            [Translation here]
            """),
            ("human", """Context: {context} \n\n Question: {question}""")
        ])

    def query(self, question: str):
        rag_chain = (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain.invoke(question)
```

## Step 4: Web Interface & API 🌐

Finally, we expose this power via a clean API using Flask. This allows our frontend application to send queries and display the bilingual results.

```python
from flask import request, jsonify
from enferno.rag.service import ArabicRAG

@rag.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    question = data.get('question')
    
    try:
        service = ArabicRAG()
        answer = service.query(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
```

## Step 5: System in Action 🚀
        
Here are examples of the system processing real-world data.

### 1. Document Ingestion Examples
The system handles various input formats, from structured scans to handwritten notes.

**Type A: Structured Document**
A standard formatted document (e.g., a menu or official form). The OCR engine identifies the layout and extracts text row-by-row.
![Structured Arabic Doc](/img/posts/arabic_menu.png)

**Type B: Handwritten/Legacy Note**
A more challenging handwritten note. The system uses advanced vision models to transcribe this into machine-readable text.
![Handwritten Arabic Note](/img/posts/example_ocr_doc.png)

### 2. PDF Chunking Example
When a large legal text is ingested, it's split into semantic chunks. Here is an example of a chunk retrieved from the vector store:

> **Chunk ID:** `doc_1024_chunk_3`
> **Source:** `Company_Articles_of_Association.pdf`
> **Content (Arabic):**
> "...يتولى مجلس الإدارة جميع السلطات اللازمة لإدارة الشركة، باستثناء ما احتفظ به القانون أو هذا النظام للجمعية العامة. يمثل رئيس المجلس الشركة أمام القضاء..."
> **Content (English Translation):**
> "...The Board of Directors shall have all powers necessary for the management of the Company, except for those reserved by law or these Articles to the General Assembly. The Chairman of the Board represents the Company before the judiciary..."

### 3. Query & Retrieval Example
The user asks a natural language question in English, and the system retrieves the relevant Arabic context to generate a bilingual answer.

**User Query:** *"What are the powers of the Board of Directors?"*

**System Response:**
```markdown
**English Answer:**
Based on the Articles of Association, the Board of Directors possesses all necessary powers to manage the company, except for specific powers explicitly reserved for the General Assembly by law or by the bylaws themselves. The Chairman also holds the authority to represent the company in legal matters.

**الإجابة بالعربية:**
بناءً على النظام الأساسي، يمتلك مجلس الإدارة كافة الصلاحيات اللازمة لإدارة الشركة، باستثناء الصلاحيات التي احتفظ بها القانون أو النظام للجمعية العامة. كما يمتلك رئيس المجلس صلاحية تمثيل الشركة أمام القضاء.
```

## Conclusion

This pipeline represents a significant step forward in making complex, multilingual datasets accessible. By combining state-of-the-art embedding models with a disciplined RAG architecture, we've created a system that doesn't just search—it understands.

Whether for legal case analysis, institutional archiving, or cross-border research, this approach ensures that language barriers don't become knowledge barriers.
