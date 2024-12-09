import os
import ollama
import fitz  # PyMuPDF để đọc PDF
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple

# Khởi tạo FastAPI app
app = FastAPI()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Thêm middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Chấp nhận tất cả các domain
    allow_credentials=True,
    allow_methods=["*"],  # Chấp nhận tất cả các phương thức HTTP
    allow_headers=["*"],  # Chấp nhận tất cả các header
)

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

VECTOR_DB = []  # Đây là cơ sở dữ liệu chứa các vector đã được tạo ra

# Đọc và trích xuất văn bản từ file PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Thêm chunk và đường dẫn vào VECTOR_DB
def add_chunk_to_database(chunk, file_path):
    embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]  # Lấy embedding của chunk
    VECTOR_DB.append((chunk, embedding, file_path))  # Lưu vào cơ sở dữ liệu

# Đọc các file PDF trong thư mục và thêm nội dung vào VECTOR_DB
pdf_directory = r"D:\Project2\data\test"  # Đường dẫn thư mục chứa file PDF
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

# Hàm retrieve sẽ trả về cả chunk và đường dẫn file
def retrieve(query, top_n=3):
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]  # Lấy embedding của câu hỏi
    similarities = []
    
    for chunk, embedding, file_path in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)  # Tính độ tương đồng cosine
        similarities.append((chunk, similarity, file_path))  # Lưu vào danh sách tương đồng
    
    similarities.sort(key=lambda x: x[1], reverse=True)  # Sắp xếp theo độ tương đồng giảm dần
    return similarities[:top_n]  # Trả về top_n kết quả có độ tương đồng cao nhất

# Cải thiện prompt chỉ dẫn
def create_instruction_prompt(retrieved_knowledge):
    instruction_prompt = f'''
    Bạn là một trợ lý thông minh, giúp tôi trả lời câu hỏi về các thông tin trong CV. 
    Dưới đây là các thông tin từ các CV mà bạn cần sử dụng để trả lời câu hỏi. Hãy chỉ sử dụng thông tin này để trả lời câu hỏi, không tạo ra thông tin mới.
    '''
    instruction_prompt += '\n'.join([f' - {chunk} (from: {file_path})' for chunk, similarity, file_path in retrieved_knowledge])  # Thêm thông tin các đoạn đã tìm thấy
    instruction_prompt += '''
    Chỉ sử dụng các thông tin trên để trả lời câu hỏi. Đừng tạo ra thông tin mới.
    '''
    return instruction_prompt

# Hàm tương tác với chatbot
def chat_with_bot(input_query, instruction_prompt):
    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[{'role': 'system', 'content': instruction_prompt},
                  {'role': 'user', 'content': input_query}],
        stream=True,
    )
    
    response = ""
    for chunk in stream:
        message_content = chunk['message']['content']
        response += message_content  # Xây dựng câu trả lời từ chatbot
    
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
    retrieved_knowledge = retrieve(input_query)
    
    # Tạo prompt chỉ dẫn cho chatbot
    instruction_prompt = create_instruction_prompt(retrieved_knowledge)
    
    # Gọi hàm để chatbot trả lời
    response = chat_with_bot(input_query, instruction_prompt)
    
    # Trả về kết quả dưới dạng JSON
    return ChatResponse(response=response)

@app.get("/pdf_info/")
async def get_pdf_info():
    # Trả về danh sách các file đã được thêm vào VECTOR_DB
    return [RetrievedKnowledge(chunk=chunk[:200], similarity=similarity, file_path=file_path)
            for chunk, similarity, file_path in VECTOR_DB]


# import os
# import ollama
# import fitz  # PyMuPDF để đọc PDF
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import List, Tuple

# # Khởi tạo FastAPI app
# app = FastAPI()

# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware

# # Thêm middleware CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Chấp nhận tất cả các domain
#     allow_credentials=True,
#     allow_methods=["*"],  # Chấp nhận tất cả các phương thức HTTP
#     allow_headers=["*"],  # Chấp nhận tất cả các header
# )




# EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
# LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

# VECTOR_DB = []  # Đây là cơ sở dữ liệu chứa các vector đã được tạo ra

# # Đọc và trích xuất văn bản từ file PDF
# def extract_text_from_pdf(pdf_path):
#     doc = fitz.open(pdf_path)
#     text = ""
#     for page in doc:
#         text += page.get_text()
#     return text

# # Thêm chunk và đường dẫn vào VECTOR_DB
# def add_chunk_to_database(chunk, file_path):
#     embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]  # Lấy embedding của chunk
#     VECTOR_DB.append((chunk, embedding, file_path))  # Lưu vào cơ sở dữ liệu

# # Đọc các file PDF trong thư mục và thêm nội dung vào VECTOR_DB
# pdf_directory = r"D:\Project2\data\test"  # Đường dẫn thư mục chứa file PDF
# for pdf_file in os.listdir(pdf_directory):
#     if pdf_file.endswith(".pdf"):
#         pdf_path = os.path.join(pdf_directory, pdf_file)
#         text = extract_text_from_pdf(pdf_path)  # Trích xuất văn bản từ PDF
#         for i, chunk in enumerate(text.split("\n")):  # Tách văn bản thành các đoạn nhỏ
#             if chunk.strip():  # Nếu dòng không rỗng
#                 add_chunk_to_database(chunk, pdf_path)  # Thêm vào VECTOR_DB
#                 print(f'Added chunk {i+1} from {pdf_file} to the database')

# # Hàm tính độ tương đồng cosine giữa hai vector
# def cosine_similarity(a, b):
#     dot_product = sum([x * y for x, y in zip(a, b)])
#     norm_a = sum([x ** 2 for x in a]) ** 0.5
#     norm_b = sum([x ** 2 for x in b]) ** 0.5
#     return dot_product / (norm_a * norm_b)  # Trả về độ tương đồng cosine

# # Hàm retrieve sẽ trả về cả chunk và đường dẫn file
# def retrieve(query, top_n=3):
#     query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]  # Lấy embedding của câu hỏi
#     similarities = []
    
#     for chunk, embedding, file_path in VECTOR_DB:
#         similarity = cosine_similarity(query_embedding, embedding)  # Tính độ tương đồng cosine
#         similarities.append((chunk, similarity, file_path))  # Lưu vào danh sách tương đồng
    
#     similarities.sort(key=lambda x: x[1], reverse=True)  # Sắp xếp theo độ tương đồng giảm dần
#     return similarities[:top_n]  # Trả về top_n kết quả có độ tương đồng cao nhất

# # Cải thiện prompt chỉ dẫn
# def create_instruction_prompt(retrieved_knowledge):
#     instruction_prompt = f'''
#     Bạn là một trợ lý thông minh, giúp tôi trả lời câu hỏi về các thông tin trong CV. 
#     Dưới đây là các thông tin từ các CV mà bạn cần sử dụng để trả lời câu hỏi. Hãy chỉ sử dụng thông tin này để trả lời câu hỏi, không tạo ra thông tin mới.
#     '''
#     instruction_prompt += '\n'.join([f' - {chunk} (from: {file_path})' for chunk, similarity, file_path in retrieved_knowledge])  # Thêm thông tin các đoạn đã tìm thấy
#     instruction_prompt += '''
#     Chỉ sử dụng các thông tin trên để trả lời câu hỏi. Đừng tạo ra thông tin mới.
#     '''
#     return instruction_prompt

# # Hàm tương tác với chatbot
# def chat_with_bot(input_query, instruction_prompt):
#     stream = ollama.chat(
#         model=LANGUAGE_MODEL,
#         messages=[{'role': 'system', 'content': instruction_prompt},
#                   {'role': 'user', 'content': input_query}],
#         stream=True,
#     )
    
#     response = ""
#     for chunk in stream:
#         message_content = chunk['message']['content']
#         response += message_content  # Xây dựng câu trả lời từ chatbot
    
#     return response

# # Pydantic model để yêu cầu dữ liệu từ người dùng
# class QueryRequest(BaseModel):
#     query: str

# class RetrievedKnowledge(BaseModel):
#     chunk: str
#     similarity: float
#     file_path: str

# class ChatResponse(BaseModel):
#     response: str

# # API endpoint cho việc nhận câu hỏi và trả về kết quả
# @app.post("/retrieve/")
# async def retrieve_information(request: QueryRequest):
#     input_query = request.query  # Nhận câu hỏi từ người dùng
#     # Lấy thông tin đã truy xuất từ cơ sở dữ liệu
#     retrieved_knowledge = retrieve(input_query)
    
#     # Tạo prompt chỉ dẫn cho chatbot
#     instruction_prompt = create_instruction_prompt(retrieved_knowledge)
    
#     # Gọi hàm để chatbot trả lời
#     response = chat_with_bot(input_query, instruction_prompt)
    
#     # Trả về kết quả dưới dạng JSON
#     return ChatResponse(response=response)

# @app.get("/pdf_info/")
# async def get_pdf_info():
#     # Trả về danh sách các file đã được thêm vào VECTOR_DB
#     return [RetrievedKnowledge(chunk=chunk[:200], similarity=similarity, file_path=file_path)
#             for chunk, similarity, file_path in VECTOR_DB]
