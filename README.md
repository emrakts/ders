# ders
s------1




!pip install groq together openai
import os
import time
from openai import OpenAI
from groq import Groq
import requests
from together import Together


# Set the API key for the OPENROUTER API
os.environ["GROQ_API_KEY"] = "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"  # Replace with your actual OpenAI key
# Set the API key for the NVIDIA API
nvidia_api_key = "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"
# Set the API key for OpenRouter API
os.environ["OPENAI_API_KEY"] = "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**"
#os.environ["OPENAI_API_KEY"] = "sk-or-v1-d28f31a29084d0a4a0f0dd6471ed3088cee2ec994bc9daf141fe85fff5c9e0**"  # Replace with your actual OpenAI key
os.environ["TOGETHER_API_KEY"] = "d66b018bf3adb4fe547aae973fe75539256c1609fa796fbe83a45ea4a40a5b09**"



'''
GROQ API
'''
# Initialize the Groq client
groq_client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Measure time for Groq API
start_time_groq = time.time()

# Create a chat completion for Groq
groq_completion = groq_client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Write e-mail regex with python",
        }
    ],
    model="llama3-8b-8192",
)

# Print the result for Groq API
print(groq_completion.choices[0].message.content)

end_time_groq = time.time()
execution_time_groq = end_time_groq - start_time_groq


'''
NVIDIA API
'''

# NVIDIA API URL'sini belirtin
nvidia_api_url = "https://integrate.api.nvidia.com/v1/chat/completions"

# Zaman ölçümünü başlat
start_time_nvidia = time.time()

# API çağrısı için istek verilerini hazırlayın
data = {
    "model": "meta/llama-3.1-70b-instruct",
    "messages": [{"role": "user", "content": "Write python regex for e-mail."}],
    "temperature": 0.2,
    "max_tokens": 1024,
    "stream": False  # Stream desteği yoksa bunu False yapın
}

# İstek başlıklarını belirtin
headers = {
    "Authorization": f"Bearer {nvidia_api_key}",
    "Content-Type": "application/json"
}

# NVIDIA API'ye POST isteği gönder
response = requests.post(nvidia_api_url, headers=headers, json=data)

# Yanıtı kontrol et ve yazdır
if response.status_code == 200:
    response_data = response.json()
    if "choices" in response_data:
        for choice in response_data["choices"]:
            print(choice["message"]["content"])
else:
    print(f"Error: {response.status_code}, {response.text}")

# Zaman ölçümünü bitir
end_time_nvidia = time.time()
execution_time_nvidia = end_time_nvidia - start_time_nvidia

'''
OPENROUTER API
'''

# Initialize OpenAI client for OpenRouter API
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENAI_API_KEY"],
)

# Measure time for OpenRouter API
start_time_openrouter = time.time()

try:
    # Create a chat completion for OpenRouter
    completion_openrouter = openrouter_client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": "YOUR_SITE_URL",  # Optional
            "X-Title": "YOUR_APP_NAME",        # Optional
        },
        model="meta-llama/llama-3.1-405b-instruct:free",
        messages=[{"role": "user", "content": "Write e-mail regex with python"}]
    )
    
    # Print the result for OpenRouter API
    print(completion_openrouter.choices[0].message.content)

except Exception as e:
    print(f"An error occurred: {e}")

end_time_openrouter = time.time()
execution_time_openrouter = end_time_openrouter - start_time_openrouter



# Initialize Together client
together_client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

# Measure time for Together API
start_time_together = time.time()
try:
    together_response = together_client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        messages=[{"role": "user", "content": "write regex email python"}],
        max_tokens=None,
        temperature=0.7,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1,
        stop=["<|eot_id|>", "<|eom_id|>"],
        safety_model="meta-llama/Meta-Llama-Guard-3-8B"
    )
    print("Together Response:", together_response.choices[0].message.content)

except Exception as e:
    print(f"Together Error: {e}")

end_time_together = time.time()
execution_time_together = end_time_together - start_time_together

# Print execution times
print(f"\nGROQ API Execution Time: {execution_time_groq:.2f} seconds")
print(f"\nTogether API Execution Time: {execution_time_together:.2f} seconds")
print(f"\nOpenRouter API Execution Time: {execution_time_openrouter:.2f} seconds")
print(f"\nNVIDIA API Execution Time: {execution_time_nvidia:.2f} seconds")




s-------2


1. Virtualbox indir ve kur ve sonrasında Extensian pack kur.
https://www.virtualbox.org/wiki/Downloads
2. Lubuntu kur. (24.04.1 veya 24.10) / iseteyenler Docker tercih edebilir.
https://lubuntu.me/downloads/ ve https://www.youtube.com/watch?v=tOfmkv_Cqww
3. Ollama sitesinden curl ile ollayı kur.
https://ollama.com/download ve curl -fsSL https://ollama.com/install.sh | sh ile
terminalde kur.
4. Ollamaya mistral modelini kur.
Lubuntu Qtterminalde ollama run mistral:latest ile kur. Terminalden soru sor cevap
almayı dene /bye ile çıkabilirsin.
5. Vscode kur.
https://code.visualstudio.com/download bu adresten .deb uzantılı paketi indir.
Muhtemelen Downloads dizini altında olur.
Terminalde cd Downlods/ ile ilgili dizine git. Daha sonra sudo dpkg -i code_1.94.2-
1728494015_amd64.deb ile kurulumu tamamla.
6. Vscode’ta herkes font size 20 yapacak.
7. Hüseyin hocanın github’taki .py dosyasını çalıştır.
https://github.com/HuseyinPARMAKSIZ/YapayZeka/blob/main/snmp-tkinter.py
Çalışmazsa terminalden sudo apt install python3 python3-pip python3-venv python3-tk
8. Bu kodta soru sayısı 2’den 5’e çıkartılacak, ayrıca veri tabanı ve web uygulama konuları
eklenecek. Ve çıktılar pdf olarak export edilmesi için bir buton eklenecek.



s------3


import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import subprocess
import re
import requests
import json

class QuestionGeneratorApp:
    def _init_(self, root):
        self.root = root
        self.root.title("Model Tabanlı Soru Üretici")
        self.root.geometry("700x700")

        # Model Seçimi Bölümü
        model_frame = ttk.LabelFrame(root, text="Modeller")
        model_frame.pack(fill="x", padx=10, pady=5)

        self.models = self.get_models()
        self.selected_model = tk.StringVar()
        if self.models:
            self.selected_model.set(self.models[0])
        else:
            self.selected_model.set("Model bulunamadı")

        self.model_dropdown = ttk.OptionMenu(model_frame, self.selected_model, *self.models)
        self.model_dropdown.pack(padx=10, pady=10)

        # Alan Seçimi Bölümü
        field_frame = ttk.LabelFrame(root, text="Alan Seçimi")
        field_frame.pack(fill="x", padx=10, pady=5)

        self.selected_field = tk.StringVar()
        fields = ["Dijital Dönüşüm", "Python Kodlama", "Power BI", "Ağ ve Güvenlik"]
        self.selected_field.set(fields[0])

        field_dropdown = ttk.OptionMenu(field_frame, self.selected_field, *fields)
        field_dropdown.pack(padx=10, pady=10)

        # Soru Üretme Butonu
        generate_button = ttk.Button(root, text="Soruları Üret", command=self.start_generation)
        generate_button.pack(pady=10)

        # Yükleme Çubuğu ve Süre
        progress_frame = ttk.Frame(root)
        progress_frame.pack(fill="x", padx=10, pady=5)

        self.progress = ttk.Progressbar(progress_frame, orient='horizontal', mode='determinate')
        self.progress.pack(fill="x", padx=10, pady=5)

        self.time_label = ttk.Label(progress_frame, text="Geçen Süre: 0s")
        self.time_label.pack(pady=5)

        # Çıktı Metin Kutusu
        output_frame = ttk.LabelFrame(root, text="Üretilen Sorular")
        output_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.output_text = tk.Text(output_frame, wrap="word")
        self.output_text.pack(fill="both", expand=True, padx=10, pady=10)

    def get_models(self):
        try:
            result = subprocess.run(['ollama', 'list'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            output = result.stdout
            models = self.parse_ollama_list(output)
            if not models:
                messagebox.showwarning("Uyarı", "Hiç model bulunamadı.")
            return models
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Hata", f"Ollama komutu çalıştırılamadı:\n{e.stderr}")
            return []
        except FileNotFoundError:
            messagebox.showerror("Hata", "Ollama komutu bulunamadı. Lütfen Ollama'nın kurulu ve PATH'e ekli olduğundan emin olun.")
            return []

    def parse_ollama_list(self, output):
        models = []
        lines = output.strip().split('\n')
        for line in lines[1:]:
            match = re.match(r'^(\S+)', line)
            if match:
                model_name = match.group(1)
                models.append(model_name)
        return models

    def start_generation(self):
        if not self.models:
            messagebox.showwarning("Uyarı", "Hiç model bulunamadı.")
            return

        selected_model = self.selected_model.get()
        selected_field = self.selected_field.get()

        threading.Thread(target=self.generate_questions, args=(selected_model, selected_field), daemon=True).start()

    def generate_questions(self, model, field):
        self.progress.config(mode='determinate', maximum=100, value=0)
        self.time_label.config(text="Geçen Süre: 0s")
        self.output_text.delete(1.0, tk.END)

        start_time = time.time()

        try:
            questions = self.create_questions_with_ollama(model, field)

            elapsed = int(time.time() - start_time)
            self.time_label.config(text=f"Geçen Süre: {elapsed}s")

            self.progress['value'] = 100

            for q in questions:
                self.output_text.insert(tk.END, f"- {q}\n")

        except Exception as e:
            self.progress['value'] = 0
            messagebox.showerror("Hata", f"Soru üretme sırasında bir hata oluştu:\n{e}")

    def create_questions_with_ollama(self, model, field):
        api_url = "http://localhost:11434/api/generate"
        payload = {
            "model": model,
            "prompt": f"Lütfen {field} alanında beş adet özgün soru üretiniz. Her bir soruyu '1.', '2.', '3.', '4.' ve '5.' ile numaralandırarak ayrı satırlarda yazınız."
        }

        headers = {
            "Content-Type": "application/json"
        }

        response = requests.post(api_url, headers=headers, data=json.dumps(payload), stream=True, timeout=60)

        if response.status_code != 200:
            raise Exception(f"API çağrısı başarısız oldu. Status Code: {response.status_code}, Mesaj: {response.text}")

        generated_text = ""
        questions = []

        try:
            for line in response.iter_lines():
                if line:
                    try:
                        json_obj = json.loads(line.decode('utf-8'))
                        response_text = json_obj.get("response", "")
                        done = json_obj.get("done", False)
                        generated_text += response_text
                        if done:
                            break
                    except json.JSONDecodeError:
                        continue

        except requests.exceptions.RequestException as e:
            raise Exception(f"API çağrısı sırasında bir hata oluştu: {e}")

        pattern = r'1\.\s*(.?)\s*2\.\s(.?)\s*3\.\s(.?)\s*4\.\s(.?)\s*5\.\s(.*)'
        match = re.search(pattern, generated_text, re.DOTALL)
        if match:
            questions = [match.group(i).strip() for i in range(1, 6)]
        else:
            questions = [q.strip() for q in generated_text.split('\n') if q.strip()]
            if len(questions) < 5:
                raise Exception("Beş soru alınamadı. API'nin yanıt formatını kontrol edin.")

        return questions[:5]
if _name_ == "_main_":
    root = tk.Tk()
    app = QuestionGeneratorApp(root)
    root.mainloop()



s-------4



do mkdir YZ3
sudo chmod 777 YZ3/
cd YZ3/
git clone https://github.com/cobanov/easy-web-summarizer.git
cd ..
python3 -m venv YZ3/
source YZ3/bin/activate
cd YZ3/
cd easy-web-summarizer/
ls -la
pip install -r requirements.txt 
python3 app/webui.py





s---------5


!pip install youtube-transcript-api
from openai import OpenAI
import re

client = OpenAI(
    base_url="https://apiv2.sooai.com.tr/openai",  # /openai ile işaretliyoruz
    api_key="sk-soo-*****"
)

from youtube_transcript_api import YouTubeTranscriptApi

'''
# Function to fetch YouTube transcript
def fetch_youtube_transcript(video_url):
    video_id = video_url.split("v=")[-1]  # Extract video ID from the URL
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    
    # Combine transcript into a single string
    transcript_text = " ".join([entry['text'] for entry in transcript])
    return transcript_text
# Fetch and print transcript
#transcript = fetch_youtube_transcript(video_url)
'''



# Fetch the Turkish transcript
# Video ID for the desired YouTube video
def extract_video_id(url):
    # Regular expression pattern for matching YouTube video IDs
    pattern = r"(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|\S+\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    
    # Search for the pattern in the URL
    match = re.search(pattern, url)
    
    if match:
        return match.group(1)  # Return the extracted video ID
    else:
        return None  # Return None if no match is found

# Example YouTube URL
video_url = "https://www.youtube.com/watch?v=bCnj14k8xWw"

# Extract video ID from the URL
video_id = extract_video_id(video_url)
print(f"Video ID: {video_id}")
transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['tr'])
print(transcript)

try:
    stream = client.chat.completions.create(
        model="default",
        messages=[
            {"role": "user", "content": f"As a professional summarizer specialized in video content, create a detailed and comprehensive summary of the YouTube video transcript: {transcript}."}
        ],
        stream=True
    )

    print("Yanıt:\n")
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end='', flush=True)
    print("\n")

except Exception as e:
    print(f"Hata oluştu: {str(e)}")



s--------6

