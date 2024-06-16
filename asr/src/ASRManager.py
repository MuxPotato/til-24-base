# Note: Code uses a finetuned Whisper Model which is too big to be uploaded to github
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import numpy as np

class ASRManager:
    def __init__(self):
        checkpoint_dir = "./model"
        self.model = WhisperForConditionalGeneration.from_pretrained(checkpoint_dir)
        self.processor = WhisperProcessor.from_pretrained(checkpoint_dir)

    def transcribe(self, audio_bytes: bytes) -> str:
        # Convert audio bytes to numpy array
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)  # assuming audio bytes are in int16 format

        # Ensure the audio array is in float32 format
        audio_array = audio_array.astype(np.float32) / np.iinfo(np.int16).max
        
        # Process the audio and prepare for transcription
        input_features = self.processor.feature_extractor(audio_array, sampling_rate=16000, return_tensors="pt").input_features
        input_features = input_features.to(self.model.device)
        
        # Perform ASR transcription with language set to English
        with torch.no_grad():
            generated_ids = self.model.generate(input_features, language='en')
            transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return transcription


