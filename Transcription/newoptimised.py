import logging
import time
import pytube
from pydub import AudioSegment
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import noisereduce as nr
import json
import os

# Set up logging
logging.basicConfig(filename='transcription.log', level=logging.INFO)

# Set device for processing
device = torch.device("cpu")
torch_dtype = torch.float32
logging.info("Using CPU for processing")

# Initialize Whisper model
model_id = "openai/whisper-large-v3"
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True).to(device)
processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor, device=device)

# Function to download audio with enhanced error handling
def download_audio(url):
    try:
        yt = pytube.YouTube(url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        if audio_stream is None:
            logging.error("No audio stream found")
            return None

        file_extension = audio_stream.mime_type.split('/')[-1]
        audio_file = audio_stream.download(filename=f'temp_audio.{file_extension}')
        full_path = os.path.abspath(audio_file)  # Get absolute path of the downloaded file
        logging.info(f"Downloaded audio file path: {full_path}")
        return full_path
    except pytube.exceptions.PytubeError as e:
        logging.error(f"Pytube error downloading audio: {e}")
    except Exception as e:
        logging.error(f"Unexpected error downloading audio: {e}")
    return None

# Function to convert audio to WAV format
def convert_to_wav(audio_path):
    try:
        if not os.path.exists(audio_path):
            logging.error(f"File not found at path: {audio_path}")
            return None

        audio = AudioSegment.from_file(audio_path)
        wav_file_path = audio_path.rsplit('.', 1)[0] + '.wav'
        audio.export(wav_file_path, format='wav')
        logging.info(f"File converted to WAV and saved at: {wav_file_path}")
        return wav_file_path
    except Exception as e:
        logging.error(f"Error converting to WAV: {e} - Path: {audio_path}")
        return None

# Function to preprocess audio
def preprocess_audio(audio_path, target_sample_rate=16000):
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)
        reduced_noise = nr.reduce_noise(audio=waveform, noise_clip=waveform[:, :5000])
        waveform = torch.from_numpy(reduced_noise)
        peak = torch.max(torch.abs(waveform))
        waveform = waveform / peak
        return waveform
    except Exception as e:
        logging.error(f"Error preprocessing audio: {e}")
    return None

# Function to transcribe audio
def transcribe(audio_path, output_format='txt'):
    try:
        logging.info(f"Transcribing audio file at path: {audio_path}")
        start_time = time.time()
        audio = {"path": audio_path}
        result = pipe(audio)
        transcription = result["text"]
        logging.info(f"Transcription completed in {time.time() - start_time:.2f} seconds")
        output_file = f'transcription.{output_format}'
        with open(output_file, 'w') as file:
            if output_format == 'txt':
                file.write(transcription)
            elif output_format == 'json':
                json.dump({"transcription": transcription}, file)
        return transcription
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
    return None

# Main function to run the script
def main(url, output_format='txt'):
    audio_file = download_audio(url)
    if audio_file:
        wav_file = convert_to_wav(audio_file)
        if wav_file:
            transcription = transcribe(wav_file, output_format)
            if transcription:
                logging.info(transcription)
                print("Transcription saved in", output_format, "format.")

# Example usage
if __name__ == "__main__":
    url = 'https://www.youtube.com/watch?v=5TXyd6jHmg0'
    main(url, 'txt')