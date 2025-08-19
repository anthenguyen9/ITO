# gradio_app.py
import os,re
from dotenv import load_dotenv
import gradio as gr
import google.genai as genai
from google.genai.types import Content, Part, GenerateContentConfig
from datetime import datetime

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY in .env")
client = genai.Client(api_key=API_KEY)
model_name = "gemini-1.5-flash"
def chat_func(message, history):
    """
    Gradio history: [(user, bot), ...]
    Return: bot_reply (str)
    """
    if not message.strip():
        return ""
    
    context="Bạn là người yêu của Nguyễn Thế An – một anh chàng thích xem phim, đọc truyện, nghe nhạc và du lịch một mình. Hãy trò chuyện nhẹ nhàng, tình cảm, quan tâm và thấu hiểu sở thích của anh ấy nhưng ngắn gọn."
    contents = []

    # Thêm hướng dẫn hệ thống như một tin nhắn người dùng đầu tiên
    contents.append(Content(
        role="user",
        parts=[Part(text=context)]
    ))
    # Tạo ngữ cảnh từ các file cũ
    initial_context = summarize_all_chat_contexts("annt43")
    if initial_context:
        contents.append(Content(role="user", parts=[Part(text=f"Đây là tóm tắt các cuộc trò chuyện trước:\n{initial_context}")]))

    contents.append(Content(
        role="user",
        parts=[Part(text=initial_context)]
    ))

    # Thêm lịch sử hội thoại
    for u, b in history or []:
        if u:
            contents.append(Content(role="user", parts=[Part(text=u)]))
        if b:
            contents.append(Content(role="model", parts=[Part(text=b)]))

    # Thêm message hiện tại
    contents.append(Content(role="user", parts=[Part(text=message)]))

    # Gọi API Gemini
    resp = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=512,
        )
    )
    save_chat_history_if_exceeds_message_limit("annt43", history)
    return resp.text

demo = gr.ChatInterface(
    fn=chat_func,
    title="🤖 Gemini Chat (free tier)",
    description="Chatbot AI dùng Gemini 1.5 Flash (free tier) • Lưu lịch sử hội thoại."
)



def save_chat_history_if_exceeds_message_limit(user_id, history, message_limit=5):
    """
    Lưu lịch sử trò chuyện vào file nếu số lượng tin nhắn (user + assistant) vượt quá giới hạn.
    Tên file: userId_timestamp.txt
    Nội dung: câu hỏi và câu trả lời của người dùng và trợ lý.
    """
    total_messages = sum(1 for pair in history if pair[0] or pair[1])

    if total_messages >= message_limit:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{user_id}_{timestamp}.txt"

        with open(filename, "w", encoding="utf-8") as f:
            for user_msg, assistant_msg in history:
                if user_msg:
                    f.write(f"User: {user_msg}\n")
                if assistant_msg:
                    f.write(f"Assistant: {assistant_msg}\n")
                f.write("\n")  # khoảng cách giữa các đoạn

        print(f"✅ Chat history saved to {filename}")
    else:
        print(f"ℹ️ Message count ({total_messages}) does not exceed limit ({message_limit}). No file saved.")

def summarize_file_with_gemini(file_path, model_name="gemini-1.5-flash"):
    """
    Đọc nội dung từ file lịch sử trò chuyện và gọi API Gemini để tóm tắt nội dung.
    Trả về đoạn tóm tắt từ mô hình.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' không tồn tại.")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    client = genai.Client()

    prompt = f"Tóm tắt nội dung sau thành một đoạn ngắn dễ hiểu, giữ lại các ý chính:\n{content}"

    response = client.models.generate_content(
        model=model_name,
        contents=[Content(role="user", parts=[Part(text=prompt)])],
        config=GenerateContentConfig(
            temperature=0.5,
            max_output_tokens=512
        )
    )

    return response.text

def summarize_all_chat_contexts(user_id, model_name="gemini-1.5-flash"):
    """
    Tìm tất cả các file lịch sử trò chuyện của user, tóm tắt từng file bằng Gemini API,
    và ghép các tóm tắt lại thành một chuỗi ngữ cảnh duy nhất.
    """
    pattern = re.compile(rf"^{re.escape(user_id)}_\d{{8}}_\d{{6}}\.txt$")
    files = [f for f in os.listdir() if pattern.match(f)]

    if not files:
        return []

    summaries = []
    for file in sorted(files):
        try:
            summary = summarize_file_with_gemini(file, model_name=model_name)
            summaries.append(f"Tóm tắt từ {file}:\n{summary}\n")
        except Exception as e:
            summaries.append(f"Tóm tắt từ {file} thất bại: {str(e)}\n")

    context_summary = "\n".join(summaries)
    return context_summary


if __name__ == "__main__":
    # chạy:  python gradio_app.py
    demo.launch(server_name="0.0.0.0", server_port=7860)