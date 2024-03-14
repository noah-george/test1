from transformers import pipeline
from .asr_interface import ASRInterface
from src.audio_utils import save_audio_to_file
import os
import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available

class WhisperASR(ASRInterface):
    def __init__(self, **kwargs):
        model_name = kwargs.get('model_name', "openai/whisper-large-v3")
        self.asr_pipeline = pipeline("automatic-speech-recognition", model=model_name, torch_dtype=torch.float16,
    device="cuda:0", # or mps for Mac devices
    model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"},)

    async def transcribe(self, client):
        file_path = await save_audio_to_file(client.scratch_buffer, client.get_file_name())
        
        if client.config['language'] is not None:
            to_return = self.asr_pipeline(file_path, generate_kwargs={"language": client.config['language']})['text']
        else:
            to_return = self.asr_pipeline(file_path)['text']

        os.remove(file_path)

        to_return = {
            "language": "UNSUPPORTED_BY_HUGGINGFACE_WHISPER",
            "language_probability": None,
            "text": to_return.strip(),
            "words": "UNSUPPORTED_BY_HUGGINGFACE_WHISPER"
        }
        return to_return
