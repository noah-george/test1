import os
from faster_whisper import WhisperModel
import ASRInterface
from src.audio_utils import save_audio_to_file
from pyannote.audio import Pipeline
import gc 
import torch

language_codes = {
    "afrikaans": "af",
    "amharic": "am",
    "arabic": "ar",
    "assamese": "as",
    "azerbaijani": "az",
    "bashkir": "ba",
    "belarusian": "be",
    "bulgarian": "bg",
    "bengali": "bn",
    "tibetan": "bo",
    "breton": "br",
    "bosnian": "bs",
    "catalan": "ca",
    "czech": "cs",
    "welsh": "cy",
    "danish": "da",
    "german": "de",
    "greek": "el",
    "english": "en",
    "spanish": "es",
    "estonian": "et",
    "basque": "eu",
    "persian": "fa",
    "finnish": "fi",
    "faroese": "fo",
    "french": "fr",
    "galician": "gl",
    "gujarati": "gu",
    "hausa": "ha",
    "hawaiian": "haw",
    "hebrew": "he",
    "hindi": "hi",
    "croatian": "hr",
    "haitian": "ht",
    "hungarian": "hu",
    "armenian": "hy",
    "indonesian": "id",
    "icelandic": "is",
    "italian": "it",
    "japanese": "ja",
    "javanese": "jw",
    "georgian": "ka",
    "kazakh": "kk",
    "khmer": "km",
    "kannada": "kn",
    "korean": "ko",
    "latin": "la",
    "luxembourgish": "lb",
    "lingala": "ln",
    "lao": "lo",
    "lithuanian": "lt",
    "latvian": "lv",
    "malagasy": "mg",
    "maori": "mi",
    "macedonian": "mk",
    "malayalam": "ml",
    "mongolian": "mn",
    "marathi": "mr",
    "malay": "ms",
    "maltese": "mt",
    "burmese": "my",
    "nepali": "ne",
    "dutch": "nl",
    "norwegian nynorsk": "nn",
    "norwegian": "no",
    "occitan": "oc",
    "punjabi": "pa",
    "polish": "pl",
    "pashto": "ps",
    "portuguese": "pt",
    "romanian": "ro",
    "russian": "ru",
    "sanskrit": "sa",
    "sindhi": "sd",
    "sinhalese": "si",
    "slovak": "sk",
    "slovenian": "sl",
    "shona": "sn",
    "somali": "so",
    "albanian": "sq",
    "serbian": "sr",
    "sundanese": "su",
    "swedish": "sv",
    "swahili": "sw",
    "tamil": "ta",
    "telugu": "te",
    "tajik": "tg",
    "thai": "th",
    "turkmen": "tk",
    "tagalog": "tl",
    "turkish": "tr",
    "tatar": "tt",
    "ukrainian": "uk",
    "urdu": "ur",
    "uzbek": "uz",
    "vietnamese": "vi",
    "yiddish": "yi",
    "yoruba": "yo",
    "chinese": "zh",
    "cantonese": "yue",
}


class FasterWhisperASR(ASRInterface):
    def __init__(self, **kwargs):
        model_size = kwargs.get('model_size', "large-v3")
        # Run on GPU with FP16
        self.asr_pipeline = WhisperModel(model_size, device="cuda", compute_type="float16")

    async def transcribe(self, client):
        file_path = await save_audio_to_file(client.scratch_buffer, client.get_file_name())

        language = None if client .config['language'] is None else language_codes.get(client.config['language'].lower())
        segments, info = self.asr_pipeline.transcribe(file_path, word_timestamps=True, language=language,vad_filter=True)
        print("hi test run case")
        segments = list(segments)  # The transcription will actually run here.
        device = "cuda" 
        
        pipeline = Pipeline.from_pretrained(
#     "pyannote/speaker-diarization-3.1",
#     use_auth_token="HUGGINGFACE_ACCESS_TOKEN_GOES_HERE")

# # send pipeline to GPU (when available)
#         import torch
#         pipeline.to(torch.device("cuda"))

# # apply pretrained pipeline
#         diarization = pipeline("audio.wav")

# # print the result
#         for turn, _, speaker in diarization.itertracks(yield_label=True):
#             print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
#         os.remove(file_path)

        

        to_return = {
        "language": info.language,
        "language_probability": info.language_probability,
        "text": ' '.join([s.text.strip() for s in result["segments"]]),
        "words":
            [
                {"word": w.word, "start": w.start, "end": w.end, "probability":w.probability} for w in flattened_words
            ]
        }
        return to_return

