import logging
import os
import time
import pytube
import torchaudio
from moviepy.editor import AudioFileClip
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import noisereduce as nr
import json
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set up logging
logging.basicConfig(filename='transcription.log', level=logging.INFO)

# Global constant for target sample rate
TARGET_SAMPLE_RATE = 16000

# Set device to CPU explicitly
device = torch.device('cpu')
logging.info(f"Using device: {device}")

# Function to download audio and convert to WAV
def download_audio(url):
    try:
        yt = pytube.YouTube(url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        if audio_stream is None:
            logging.error("No audio stream found")
            return None

        temp_video_file = audio_stream.download(filename='temp_video')
        logging.info(f"Downloaded video file path: {temp_video_file}")

        if os.path.exists(temp_video_file):
            # Convert to WAV using moviepy
            audio_clip = AudioFileClip(temp_video_file)
            wav_file_path = temp_video_file.rsplit('.', 1)[0] + '.wav'
            audio_clip.write_audiofile(wav_file_path)
            audio_clip.close()

            # Remove the temporary video file
            os.remove(temp_video_file)

            logging.info(f"Converted to WAV and saved at: {wav_file_path}")
            return wav_file_path if os.path.exists(wav_file_path) else None
        else:
            logging.error(f"Downloaded file not found at path: {temp_video_file}")
            return None

    except Exception as e:
        logging.error(f"Error in downloading and converting audio: {e}")
        return None

# Function to preprocess audio with noise reduction and normalization
def preprocess_audio(audio_path, target_sample_rate=16000):
    try:
        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert to single channel (mono) if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if necessary
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)

        # Convert waveform to numpy array for noisereduce library
        waveform_np = waveform.numpy()

        # Apply noise reduction
        reduced_noise = nr.reduce_noise(y=waveform_np, sr=sample_rate)

        # Convert back to PyTorch tensor
        waveform = torch.from_numpy(reduced_noise)

        # Normalize audio using peak normalization
        peak = torch.max(torch.abs(waveform))
        waveform = waveform / peak

        return waveform
    except Exception as e:
        logging.error(f"Error preprocessing audio: {e}")
    return None

# Function to transcribe audio with Whisper model
def transcribe(audio_path, model_name, batch_size=10, output_format='txt'):
    try:
        start_time = time.time()

        # Load the model and processor
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name).to(device)
        model.eval()

        # Set up the pipeline
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device,
        )

        # Load and preprocess the audio
        waveform = preprocess_audio(audio_path, TARGET_SAMPLE_RATE)
        if waveform is None:
            return None

        # Convert the waveform back to numpy array for the pipeline
        waveform_np = waveform.squeeze().numpy()

        # Transcribe using the pipeline
        result = pipe(waveform_np, return_timestamps=True)
        transcription = result['text']

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
    transcription = transcribe(audio_file, model_name, batch_size, output_format)
    if transcription:
        logging.info(transcription)
        print("Transcription saved in", output_format, "format.")
