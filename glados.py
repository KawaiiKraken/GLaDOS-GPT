#!/bin/python3
import os
import signal
import subprocess 
import whisper
import openai
import random
import torch
from utils.tools import prepare_text
from scipy.io.wavfile import write
import time
from sys import modules as mod
from subprocess import call
import pvporcupine
import struct
import pyaudio
from rhasspysilence import WebRtcVadRecorder,VoiceCommand, VoiceCommandResult
import threading
import dataclasses
import typing
from queue import Queue
import json
import io
from pathlib import Path
import shlex
import wave
import sys



def process_args():
    args = sys.argv[1:] # exclude the script name from the arguments
    
    if "--help" in args or "-h" in args:
        print("Usage: glados.py [options]\n"
              "Options:\n"
              "  --model [model]     Sets whisper model. Default is medium(+.en).\n"
              "  --no-gpt            Uses 'this is a test' instead of GPT for answers.\n"
              "  --confirm           Asks to verify input (WiP).\n"
              "  --no-voicelines     Skips all game voicelines.\n"
              "  --help | -h         Prints this message.\n")
        exit()
    
    whisper_model = "medium"
    use_gpt = True
    confirm_input = False
    voicelines = True
    
    for i, arg in enumerate(args):
        if arg == "--model":
            whisper_model = args[i+1]
        elif arg == "--no-gpt":
            use_gpt = False
        elif arg == "--confirm":
            confirm_input = True
        elif arg == "--no-voicelines":
            voicelines = False
    
    return whisper_model, use_gpt, confirm_input, voicelines



def prepare_gpt():
    openai.api_key = os.environ.get('OPENAI_API_KEY')

    # make gpt act as glados
    conversation_history = "User: act as GLaDOS from portal. Be snarky and try to poke jokes at the user when possible. When refering to the User use words like Chell, she, you. Keep the responses as short as possible without breaking character."
    return conversation_history



def speech_to_text(stt_model, input_filename):
    # if recording stops too early or late mess with vad_mode sample_rate and silence_seconds
    pa = pyaudio.PyAudio()
    vad_mode = 3
    sample_rate = 16000
    min_seconds = 1
    max_seconds = 48
    speech_seconds = 0.1
    silence_seconds = 0.3
    before_seconds = 0.2
    chunk_size= 960
    skip_seconds = 0
    audio_source = None
    channels = 1
    recorder = WebRtcVadRecorder(
        vad_mode=vad_mode,
        sample_rate=48000,
        min_seconds=min_seconds,
        max_seconds=max_seconds,
        speech_seconds=speech_seconds,
        silence_seconds=silence_seconds,
        before_seconds=before_seconds,
    )

    recorder.start()
    wav_sink = 'wavs/'
    wav_dir = None
    wav_filename = input_filename
    if wav_sink:
        wav_sink_path = Path(wav_sink)
        if wav_sink_path.is_dir():
            wav_dir = wav_sink_path
        else:
            wav_sink = open(wav_sink, "wb")
    voice_command: typing.Optional[VoiceCommand] = None
    audio_source = pa.open(rate=sample_rate,format=pyaudio.paInt16,channels=channels,input=True,frames_per_buffer=chunk_size)
    audio_source.start_stream()
    print("Recording...", file=sys.stderr)
    def buffer_to_wav(buffer: bytes) -> bytes:
        """Wraps a buffer of raw audio data in a WAV"""
        rate = int(sample_rate)
        width = int(2)
        channels = int(1)
 
        with io.BytesIO() as wav_buffer:
            wav_file: wave.Wave_write = wave.open(wav_buffer, mode="wb")
            with wav_file:
                wav_file.setframerate(rate)
                wav_file.setsampwidth(width)
                wav_file.setnchannels(channels)
                wav_file.writeframesraw(buffer)
 
            return wav_buffer.getvalue()
    try:
        chunk = audio_source.read(chunk_size)
        while chunk:
 
            # Look for speech/silence
            voice_command = recorder.process_chunk(chunk)
 
            if voice_command:
                is_timeout = voice_command.result == VoiceCommandResult.FAILURE
                # Reset
                audio_data = recorder.stop()
                if wav_dir:
                    # Write WAV to directory
                    wav_path = (wav_dir / time.strftime(wav_filename)).with_suffix(
                        ".wav"
                    )
                    wav_bytes = buffer_to_wav(audio_data)
                    wav_path.write_bytes(wav_bytes)
                    print('file saved')
                    break
                elif wav_sink:
                    # Write to WAV file
                    wav_bytes = core.buffer_to_wav(audio_data)
                    wav_sink.write(wav_bytes)
            # Next audio chunk
            chunk = audio_source.read(chunk_size)
 
    finally:
        try:
            audio_source.close_stream()
        except Exception:
            pass
    audio = whisper.load_audio(wav_path)
    audio= whisper.pad_or_trim(audio)
    result = stt_model.transcribe(audio)
    return result["text"] # return transcript 



def load_tts():
    # Select the device
    if torch.is_vulkan_available():
        device = 'vulkan'
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # Load models
    glados = torch.jit.load('models/glados.pt')
    vocoder = torch.jit.load('models/vocoder-gpu.pt', map_location=device)

    # Prepare models in RAM
    for i in range(2):
        init = glados.generate_jit(prepare_text(str(i)))
        init_mel = init['mel_post'].to(device)
        init_vo = vocoder(init_mel)
    return glados, vocoder, device



def tts(text, glados, vocoder, device):
    # Tokenize, clean and phonemize input text
    x = prepare_text(text).to('cpu')

    with torch.no_grad():

        # Generate generic TTS-output
        old_time = time.time()
        tts_output = glados.generate_jit(x)
        # print("Forward Tacotron took " + str((time.time() - old_time) * 1000) + "ms") # debug 

        # Use HiFiGAN as vocoder to make output sound like GLaDOS
        old_time = time.time()
        mel = tts_output['mel_post'].to(device)
        audio = vocoder(mel)
        # print("HiFiGAN took " + str((time.time() - old_time) * 1000) + "ms") # debug
        
        # Normalize audio to fit in wav-file
        audio = audio.squeeze()
        audio = audio * 32768.0
        audio = audio.cpu().numpy().astype('int16')
        output_file = ('wavs/output.wav')
        
        # Write audio file to disk
        # 22,05 kHz sample rate
        write(output_file, 22050, audio)

        # Play audio file
        call(["mpv", "wavs/output.wav", "--no-terminal"])



def detect_keyword(stt_model, conversation_history, glados, vocoder, device, use_gpt, confirm_input):
    print("\nlistening for keyword...")
    porcupine = None
    pa = None
    audio_stream = None

    try:
        porcu_key = os.environ.get('PICOVOICE_KEY')
        porcupine = pvporcupine.create(
            access_key=porcu_key,
            keyword_paths=['models/hey-glad-os_en_linux_v2_2_0.ppn']
        )

        pa = pyaudio.PyAudio()

        audio_stream = pa.open(
                        rate=porcupine.sample_rate,
                        channels=1,
                        format=pyaudio.paInt16,
                        input=True,
                        frames_per_buffer=porcupine.frame_length)


        while True:
            pcm = audio_stream.read(porcupine.frame_length)
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

            keyword_index = porcupine.process(pcm)

            if keyword_index >= 0:
                print("Wake-Word Detected")
                print("conversation_loop()")
                conversation_loop(stt_model, conversation_history, glados, vocoder, device, use_gpt, confirm_input)
    finally:
        if porcupine is not None:
            porcupine.delete()

        if audio_stream is not None:
            audio_stream.close()

        if pa is not None:
            pa.terminate()



def conversation_loop(stt_model, conversation_history, glados, vocoder, device, use_gpt, confirm_input):
    # Continue the conversation
    if True:
        try: # try loop to get a cool message on ctrl+c
            # Get user input
            print()
            input_filename = "input"
            user_input = speech_to_text(stt_model, input_filename)
            print("Chell: " + user_input)
            # user_input = "test" # text not voice
            # selection = input("\nis this satisfactory? [y]/[n]") # text not voice
            selection = ""
            if (confirm_input == True):
                selection_filename = "selection"
                selection = speech_to_text(stt_model, selection_filename)
                selection = selection.lower()
                print("\nSelection: " + selection)
            if ( (confirm_input == False) | ("yes" in selection) | ("yeah" in selection) ):
                # Add the user's input to the conversation history
                conversation_history += "\nUser: " + user_input
                prompt = conversation_history + user_input

                # Generate a response based on the conversation history
                if (use_gpt == True):
                    full_response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[ {"role": "system", "content": prompt} ], temperature=0, max_tokens=100)
                    # Extract the response text from the API response
                    message = full_response.choices[0].message.content.strip()
                if (use_gpt == False):
                    message = "this is a test"

                # Add the response to the conversation history
                conversation_history += "\nChatGPT: " + message
                print()
                print("GLaDOS: ", message)
                tts(message, glados, vocoder, device)
            print("listening for keyword...")
        except KeyboardInterrupt:
            print()
            print("\nglados.goodbye()")
            if voicelines == True:
                os.system("mpv sounds/exit_messages/" + random.choice(os.listdir("sounds/exit_messages/")) + " --no-terminal") 
            exit()



def main():
    whisper_model, use_gpt, confirm_input, voicelines = process_args()
    print("Config:",
          "\n  whisper_model: " + whisper_model + ".en",
          "\n  use_gpt: " + str(use_gpt),
          "\n  confirm_input: " + str(confirm_input),
          "\n  voicelines: " + str(voicelines) + "\n")
    conversation_history = prepare_gpt()
    print("announce.powerup.init()")
    if voicelines == True:
        os.system("mpv sounds/Announcer_wakeup_powerup01.wav --no-terminal")
    print("loading stt_model...")
    stt_model = whisper.load_model(whisper_model + ".en")
    print("stt_model loaded")
    print("loading tts...")
    glados, vocoder, device = load_tts()
    print("tts loaded")
    print("announce.powerup.complete()")
    if voicelines == True:
        os.system("mpv sounds/Announcer_wakeup_powerup02.wav --no-terminal") # powerup 
    # i found these to be annoying but you can enable them
    # print("glados.hello()")
    # if voicelines == True:
        # os.system("mpv sounds/welcome_messages/" + random.choice(os.listdir("sounds/welcome_messages/")) + " --no-terminal") 
    detect_keyword(stt_model, conversation_history, glados, vocoder, device, use_gpt, confirm_input)



if __name__ == "__main__":
    main()
