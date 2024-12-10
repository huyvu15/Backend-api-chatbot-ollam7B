import os
import glob
import PyPDF2
import requests
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


from fastapi.middleware.cors import CORSMiddleware


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:5173",  # URL của frontend của bạn
    "http://localhost",
    # Thêm các nguồn cần thiết khác
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Cho phép frontend của bạn
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class QueryRequest(BaseModel):
    query: str


class RetrievedKnowledge(BaseModel):
    chunk: str
    similarity: float
    file_path: str

class ChatResponse(BaseModel):
    response: str

# Các hàm xử lý giống như trước
def load_pdfs(folder_path: str) -> List[Dict[str, Any]]:
    pdf_docs = []
    unique_files = set()
    for file in glob.glob(os.path.join(folder_path, "*.pdf")):
        try:
            if file in unique_files:
                continue
            unique_files.add(file)
            
            with open(file, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
                pdf_docs.append({
                    "text": text,
                    "file_path": file
                })
        except Exception as e:
            print(f"Lỗi xử lý tệp {file}: {e}")
    return pdf_docs


def chunk_documents(pdf_docs: List[Dict[str, Any]], chunk_size=500, chunk_overlap=100) :#-> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    documents = []
    for doc in pdf_docs:
        chunks = text_splitter.split_text(doc["text"])
        for chunk in chunks:
            documents.append(Document(
                page_content=chunk, 
                metadata={
                    "file_path": doc["file_path"], 
                    "full_text": doc["text"]
                }
            ))
    return documents


def create_vectorstore(documents: List[Document], embedding_model: HuggingFaceEmbeddings) -> FAISS:
    return FAISS.from_documents(documents, embedding_model)


def search_candidates(query: str, vectorstore: FAISS, top_k: int = 2):#-> List[Dict[str, str]]:
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    # results = retriever.get_relevant_documents(query)
    results = retriever.invoke(query)

    
    unique_results = []
    seen_paths = set()
    for doc in results:
        if doc.metadata["file_path"] not in seen_paths:
            unique_results.append({
                "file_path": doc.metadata["file_path"], 
                "context": doc.metadata["full_text"]
            })
            seen_paths.add(doc.metadata["file_path"])
    
    return unique_results


def generate_response(query: str, context: str, together_api_key: str, together_model: str):# -> str:
    if not together_api_key:
        raise ValueError("Khóa API Together.ai là bắt buộc")
    
    headers = {
        "Authorization": f"Bearer {together_api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": together_model,
        "prompt": f"""Bạn là một chatbot AI có khả năng tìm kiếm và đưa ra thông tin của ứng viên phù hợp, dựa trên toàn bộ bối cảnh CV ứng viên được tìm thấy sau đây, hãy đưa ra phân tích và trả lời ý kiến của bạn câu truy vấn một cách chi tiết và chính xác:
    
        Bối cảnh CV: {context}
        Câu truy vấn: {query}

        Trả lời chi tiết bằng tiếng Việt:""",
        "max_tokens": 500,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(
            "https://api.together.xyz/v1/completions", 
            headers=headers, 
            json=payload
        )
        
        response_data = response.json()
        # print(response_data)
        if 'choices' in response_data and len(response_data['choices']) > 0:
            print(response_data['choices'][0]['text'].strip())
            return response_data['choices'][0]['text'].strip()
        elif 'output' in response_data:
            print(response_data['output'].strip())
            return response_data['output'].strip()
        else:
            return "Không tìm thấy thông tin phù hợp với truy vấn."
    
    except requests.RequestException as e:
        return f"Lỗi kết nối: {e}"
    except Exception as e:
        return f"Lỗi không xác định: {e}"


def process_search(query: str, vectorstore: FAISS, together_api_key: str, together_model: str):# -> List[Dict[str, str]]:
    matched_candidates = search_candidates(query, vectorstore)
    
    results = []
    for candidate in matched_candidates:
        try:
            response = generate_response(query, candidate['context'], together_api_key, together_model)
            
            results.append({
                "file_path": candidate['file_path'],
                "response": response + f"\n\n🔗 Đường dẫn CV: {candidate['file_path']}"
            })
        except Exception as e:
            print(f"Lỗi xử lý tệp {candidate['file_path']}: {e}")
    
    return results


# Pydantic model for input request
class QueryRequest(BaseModel):
    query: str


# API endpoints
@app.post("/search")
async def search(query_request: QueryRequest):
    query = query_request.query
    folder_path = "D:\\Project2\\data\\test"
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True})
    together_api_key = "eca0b727abc5861fdcb4ea8bfcad9e1c165fd552cf1b70859350cad33ba8e15d"
    # together_model = "meta-llama/Llama-2-7b-chat-hf"
    together_model = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    
    
    # Load and process documents
    pdf_docs = load_pdfs(folder_path)
    documents = chunk_documents(pdf_docs)
    vectorstore = create_vectorstore(documents, embedding_model)
    
    # Process search
    results = process_search(query, vectorstore, together_api_key, together_model)
    
    if results:
        return {"results": results}
    else:
        raise HTTPException(status_code=404, detail="Không tìm thấy kết quả phù hợp.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)