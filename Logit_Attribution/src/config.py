from dotenv import load_dotenv
import os
load_dotenv()
hugging_face_token = os.getenv('hugging_face_token')

from huggingface_hub import login
login(token= hugging_face_token)