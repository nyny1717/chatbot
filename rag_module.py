import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# .env 파일에 저장된 API 키 로드 
load_dotenv()

def create_rag_chain(pdf_path):
    # [1단계] 문서 로드 (Document Load)
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()

    # [2단계] 문서 분할 (Text Split)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,         # 청크 사이즈 조절
        chunk_overlap=100,      # 오버랩 조절
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    split_documents = text_splitter.split_documents(docs)

    # [3~4단계] 임베딩 및 벡터 DB 저장 (Google Embedding으로 변경)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
    
    # [5단계] 검색기(Retriever) 생성
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # [6~7단계] 프롬프트 및 LLM 설정
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}

    Answer in Korean:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Google Gemini 모델로 변경
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        timeout=20,
        max_retries=1
    )

    # [8단계] 체인 생성 (Chain)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain