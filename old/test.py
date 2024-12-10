import os
import pickle
import ollama
import fitz  # PyMuPDF để đọc PDF
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple
import PyPDF2
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# Khởi tạo FastAPI app
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

# Thêm middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Chấp nhận tất cả các domain
    allow_credentials=True,
    allow_methods=["*"],  # Chấp nhận tất cả các phương thức HTTP
    allow_headers=["*"],  # Chấp nhận tất cả các header
)

# Đường dẫn đến file lưu VECTOR_DB
VECTOR_DB_FILE = 'vector_db.pkl'

VECTOR_DB = []  # Đây là cơ sở dữ liệu chứa các vector đã được tạo ra

# Lưu VECTOR_DB vào file pickle
def save_vector_db():
    with open(VECTOR_DB_FILE, 'wb') as f:
        pickle.dump(VECTOR_DB, f)

# Tải VECTOR_DB từ file pickle
def load_vector_db():
    global VECTOR_DB
    if os.path.exists(VECTOR_DB_FILE):
        with open(VECTOR_DB_FILE, 'rb') as f:
            VECTOR_DB = pickle.load(f)
    else:
        VECTOR_DB = []  # Nếu không có file, khởi tạo VECTOR_DB rỗng

# Đọc và trích xuất văn bản từ file PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Hàm load mô hình embedding
def load_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"):
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True}  # Đảm bảo tính năng cosine similarity
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    return embedding_model

# Hàm tạo FAISS vectorstore từ tài liệu và embeddings
def generate_embeddings(docs, embedding_model):
    db = FAISS.from_documents(documents=docs, embedding=embedding_model)
    return db

# Thêm chunk và đường dẫn vào VECTOR_DB
def add_chunk_to_database(chunk, file_path):
    embedding = ollama.embed(model="hf.co/CompendiumLabs/bge-base-en-v1.5-gguf", input=chunk)['embeddings'][0]  # Lấy embedding của chunk
    VECTOR_DB.append((chunk, embedding, file_path))  # Lưu vào cơ sở dữ liệu

# Đọc các file PDF trong thư mục và thêm nội dung vào VECTOR_DB
def process_pdfs(pdf_directory):
    for pdf_file in os.listdir(pdf_directory):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, pdf_file)
            text = extract_text_from_pdf(pdf_path)  # Trích xuất văn bản từ PDF
            for i, chunk in enumerate(text.split("\n")):  # Tách văn bản thành các đoạn nhỏ
                if chunk.strip():  # Nếu dòng không rỗng
                    add_chunk_to_database(chunk, pdf_path)  # Thêm vào VECTOR_DB
                    print(f'Added chunk {i+1} from {pdf_file} to the database')

# Hàm tính độ tương đồng cosine giữa hai vector
def cosine_similarity(a, b):
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    return dot_product / (norm_a * norm_b)  # Trả về độ tương đồng cosine

# Hàm retrieve sẽ trả về cả nội dung của file và đường dẫn file
# Hàm retrieve sẽ trả về cả nội dung của file và đường dẫn file
def retrieve(query, top_n=5):
    embedding_model = load_embedding_model()  # Load mô hình embedding
    query_embedding = ollama.embed(model="hf.co/CompendiumLabs/bge-base-en-v1.5-gguf", input=query)['embeddings'][0]  # Lấy embedding của câu hỏi
    similarities = []
    
    # Lặp qua VECTOR_DB để tính độ tương đồng với từng chunk
    for chunk, embedding, file_path in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)  # Tính độ tương đồng cosine
        similarities.append((chunk, similarity, file_path))  # Lưu vào danh sách tương đồng
    
    # Sắp xếp các chunk theo độ tương đồng giảm dần
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Lấy các chunk thuộc về file có độ tương đồng cao nhất
    top_file_path = similarities[0][2]  # File có độ tương đồng cao nhất
    top_chunks = [chunk for chunk, _, file_path in similarities if file_path == top_file_path][:top_n]  # Chỉ lấy chunk của file đó

    # In ra các chunk để kiểm tra trong terminal
    print(f"Top {top_n} chunks for query '{query}' from file {top_file_path}:")
    for i, chunk in enumerate(top_chunks, 1):
        print(f"Chunk {i}: {chunk}\n")
    
    return top_chunks, top_file_path  # Trả về 5 chunk đầu tiên và đường dẫn file


# Cải thiện prompt chỉ dẫn
def create_instruction_prompt(top_chunks, file_path):
    instruction_prompt = f'''
    Bạn là một trợ lý thông minh, giúp tôi tìm kiếm và trả lời các thông tin được cung cấp. 
    Dưới đây là các thông tin từ các tài liệu mà bạn cần sử dụng để trả lời câu hỏi. Hãy chỉ sử dụng thông tin này để trả lời câu hỏi, không tạo ra thông tin mới.
    '''
    
    # Thêm thông tin về 5 chunk đầu tiên của file có độ tương đồng cao nhất
    for chunk in top_chunks:
        instruction_prompt += f'\n - {chunk} (file: {file_path})'

    instruction_prompt += '''
    Chỉ sử dụng các thông tin trên để trả lời câu hỏi. Đừng tạo ra thông tin mới.
    '''
    
    return instruction_prompt

# Hàm tương tác với chatbot
def chat_with_bot(input_query, instruction_prompt, retrieved_knowledge):
    # Gửi câu hỏi và các thông tin đã truy xuất cho chatbot
    stream = ollama.chat(
        model='hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF',
        messages=[{'role': 'system', 'content': instruction_prompt},
                  {'role': 'user', 'content': input_query}],
        stream=True,
    )
    
    response = ""
    for chunk in stream:
        message_content = chunk['message']['content']
        response += message_content  # Xây dựng câu trả lời từ chatbot
    
    # Sau khi chatbot trả lời, thêm đường dẫn của file vào câu trả lời
    response += "\n\nTham khảo từ tài liệu sau:"
    response += f"\n- {retrieved_knowledge[1]}"  # Chỉ hiển thị đường dẫn file có độ tương đồng cao nhất

    return response

# Pydantic model để yêu cầu dữ liệu từ người dùng
class QueryRequest(BaseModel):
    query: str

class RetrievedKnowledge(BaseModel):
    chunk: str
    similarity: float
    file_path: str

class ChatResponse(BaseModel):
    response: str

# API endpoint cho việc nhận câu hỏi và trả về kết quả
@app.post("/retrieve/")  
async def retrieve_information(request: QueryRequest):
    input_query = request.query  # Nhận câu hỏi từ người dùng
    # Lấy thông tin đã truy xuất từ cơ sở dữ liệu
    top_chunks, top_file_path = retrieve(input_query)
    
    # Tạo prompt chỉ dẫn cho chatbot
    instruction_prompt = create_instruction_prompt(top_chunks, top_file_path)
    
    # Gọi hàm để chatbot trả lời
    response = chat_with_bot(input_query, instruction_prompt, (top_chunks, top_file_path))
    
    # Trả về kết quả dưới dạng JSON
    return ChatResponse(response=response)

@app.get("/pdf_info/") 
async def get_pdf_info():
    # Trả về danh sách các file đã được thêm vào VECTOR_DB
        return [RetrievedKnowledge(chunk=chunk[:200], similarity=similarity, file_path=file_path) 
            for chunk, similarity, file_path in VECTOR_DB]

# Tải dữ liệu vector khi khởi tạo ứng dụng
load_vector_db()

# Đường dẫn thư mục chứa file PDF của bạn
pdf_directory = r"D:\Project2\data\test"
process_pdfs(pdf_directory)  # Xử lý các PDF và lưu dữ liệu vào VECTOR_DB
