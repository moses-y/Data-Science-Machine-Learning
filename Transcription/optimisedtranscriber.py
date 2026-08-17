import logging
import time
#import numpy as np
import pytube
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import noisereduce as nr
#from tqdm import tqdm
import json
#from pydub import AudioSegment
import os
#from faster_whisper import WhisperModel

# Set up logging
logging.basicConfig(filename='transcription.log', level=logging.INFO)

# Set device to CPU
device = torch.device("cpu")
logging.info("Using CPU for processing")

# Initialize Whisper model
device = torch.device("cpu")
torch_dtype = torch.float32
model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
).to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=device,
)
# Function to download audio with enhanced error handling
def download_audio(url):
    try:
        yt = pytube.YouTube(url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        audio_file = audio_stream.download(filename='temp_audio')  # Keep the filename without extension

        logging.info(f"Downloaded audio file path: {audio_file}")
        if os.path.exists(audio_file):
            logging.info("File exists and is ready for conversion.")
        else:
            logging.error("Downloaded file does not exist.")
            return None
        
        return audio_file
    except pytube.exceptions.PytubeError as e:
        logging.error(f"Pytube error downloading audio: {e}")
    except Exception as e:
        logging.error(f"Unexpected error downloading audio: {e}")
        return None

def convert_to_wav(audio_path):
    try:
        logging.info(f"Converting file at path: {audio_path}")

        # Define the output path for the WAV file
        wav_file_path = audio_path.rsplit('.', 1)[0] + '.wav'

        # Load the downloaded file with torchaudio
        waveform, sample_rate = torchaudio.load(audio_path)

        # Export as WAV using torchaudio
        torchaudio.save(wav_file_path, waveform, sample_rate)

        logging.info(f"File converted to WAV and saved at: {wav_file_path}")
        return wav_file_path
    except Exception as e:
        logging.error(f"Error converting to WAV: {e}")
        return None

# Function to preprocess audio with noise reduction and normalization
def preprocess_audio(audio_path, target_sample_rate=16000):
    try:
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample if necessary
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)

        # Apply noise reduction
        reduced_noise = nr.reduce_noise(audio=waveform, noise_clip=waveform[:, :5000])
        waveform = torch.from_numpy(reduced_noise)

        # Normalize audio using peak normalization
        peak = torch.max(torch.abs(waveform))
        waveform = waveform / peak

        return waveform
    except Exception as e:
        logging.error(f"Error preprocessing audio: {e}")
    return None

# Function to transcribe audio with CPU optimization
def transcribe(audio_path, output_format='txt'):
    try:
        logging.info(f"Transcribing audio file at path: {audio_path}")
        start_time = time.time()

        # Transcribe using Whisper pipeline
        audio = {"path": audio_path}
        result = pipe(audio)
        transcription = result["text"]

        logging.info(f"Transcription completed in {time.time() - start_time:.2f} seconds")

        # Save the transcription
        if output_format == 'txt':
            with open('transcription.txt', 'w') as file:
                file.write(transcription)
        elif output_format == 'json':
            with open('transcription.json', 'w') as file:
                json.dump({"transcription": transcription}, file)

        return transcription
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        return None

# Example usage with customization options

url = 'https://youtu.be/9dSkvxS2EB0'
model_name = 'openai/whisper-large-v3'
batch_size = 10
output_format = 'txt'

audio_file = download_audio(url)
if audio_file:
    wav_file = convert_to_wav(audio_file)
    if wav_file:
        transcription = transcribe(wav_file, model_name, batch_size, output_format)
        if transcription:
            logging.info(transcription)
            print("Transcription saved in", output_format, "format.")