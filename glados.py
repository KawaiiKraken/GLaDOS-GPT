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
import nltk
import numpy as np


nltk.download('punkt')

def process_args():
    args = sys.argv[1:] # exclude the script name from the arguments
    
    if "--help" in args or "-h" in args:
        print("Usage: glados.py [options]\n"
              "Options:\nn"
              "  --model [model]     Sets whisper model. Default is medium(+.en).\n"
              "  --no-gpt            Uses 'this is a test' instead of GPT for answers.\n"
              "  --confirm           Asks to verify input (WiP).\n"
              "  --no-voicelines     Skips all game voicelines.\n"
              "  --save              Saves the current conversation.\n"
              "  --load              Loads a saved conversation.\n"
              "  --no-tts            Text only responses.\n"
              "  --no-stt            Type commands instead.\n"
              "  --help | -h         Prints this message.\n")
        exit()

    global whisper_model 
    global use_gpt 
    global confirm_input 
    global voicelines 
    global convo_save
    global convo_load 
    global stt_enabled 
    global tts_enabled  
    
    whisper_model = "medium"
    use_gpt = True
    confirm_input = False
    voicelines = True
    convo_save = False
    convo_load = False
    tts_enabled = True
    stt_enabled = True
    
    for i, arg in enumerate(args):
        if arg == "--model":
            whisper_model = args[i+1]
        elif arg == "--no-gpt":
            use_gpt = False
        elif arg == "--confirm":
            confirm_input = True
        elif arg == "--no-voicelines":
            voicelines = False
        elif arg == "--save":
            convo_save = True
        elif arg == "--load":
            convo_load = True
        elif arg == "--no-tts":
            tts_enabled = False
        elif arg == "--no-stt":
            stt_enabled = False

    print("Config:",
        "\n  whisper_model: " + whisper_model + ".en",
        "\n  use_gpt: " + str(use_gpt),
        "\n  confirm_input: " + str(confirm_input),
        "\n  voicelines: " + str(voicelines),
        "\n  convo_save: " + str(convo_save),
        "\n  convo_load: " + str(convo_load),
        "\n  stt: " + str(stt_enabled),
        "\n  tts: " + str(tts_enabled),
        "\n")

    
    return whisper_model 


openai.api_key = os.environ.get('OPENAI_API_KEY')

def buffer_to_wav(buffer: bytes) -> bytes:
    """Wraps a buffer of raw audio data in a WAV"""
    global sample_rate
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


def speech_to_text(stt_model):
    # if recording stops too early or late mess with vad_mode sample_rate and silence_seconds
    pa = pyaudio.PyAudio()
    vad_mode = 3
    global sample_rate
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
    voice_command: typing.Optional[VoiceCommand] = None
    audio_source = pa.open(rate=sample_rate,format=pyaudio.paInt16,channels=channels,input=True,frames_per_buffer=chunk_size)
    audio_source.start_stream()
    print("Recording...", file=sys.stderr)
    try:
        chunk = audio_source.read(chunk_size)
        while chunk:
 
            # Look for speech/silence
            voice_command = recorder.process_chunk(chunk)
 
            if voice_command:
                is_timeout = voice_command.result == VoiceCommandResult.FAILURE
                # Reset
                audio_data = recorder.stop()
                print('file saved')
                break
            # Next audio chunk
            chunk = audio_source.read(chunk_size)
 
    finally:
        try:
            audio_source.close_stream()
        except Exception:
            pass
    audio = np.frombuffer(audio_data, np.int16).astype(np.float32)*(1/32768.0)
    audio = whisper.pad_or_trim(audio)
    transcription = stt_model.transcribe(audio)
    return transcription["text"] # return transcript 



def load_tts():
    global glados_voice, vocoder, device
    # Select the device
    if torch.is_vulkan_available():
        device = 'vulkan'
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # Load models
    glados_voice = torch.jit.load('models/glados.pt')
    vocoder = torch.jit.load('models/vocoder-gpu.pt', map_location=device)

    # Prepare models in RAM
    for i in range(2):
        init = glados_voice.generate_jit(prepare_text(str(i)))
        init_mel = init['mel_post'].to(device)
        init_vo = vocoder(init_mel)




def tts(text):
    global glados_voice, vocoder, device
    # Tokenize, clean and phonemize input text
    x = prepare_text(text).to('cpu')

    with torch.no_grad():

        # Generate generic TTS-output
        old_time = time.time()
        tts_output = glados_voice.generate_jit(x)
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



def detect_keyword():
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
                return 
    finally:
        if porcupine is not None:
            porcupine.delete()

        if audio_stream is not None:
            audio_stream.close()

        if pa is not None:
            pa.terminate()


def count_tokens(text):
    if text != None:
        tokens = nltk.word_tokenize(text)
        num_tokens = len(tokens)
        return num_tokens


#make gpt act as glados
conversation_history = "User: act as GLaDOS from portal. Be snarky and try to poke jokes at the user when possible. When refering to the User use the name Chell. Keep the responses as short as possible without breaking character."
convo_loaded = False

def conversation_loop(stt_model=None):
    try: # try loop to get a cool message on ctrl+c
        # Get user input
        print()
        if stt_enabled == True:
            user_input = speech_to_text(stt_model)
            print("Chell: " + user_input)
        else: 
            user_input = input("Chell: ")
        # selection = input("\nis this satisfactory? [y]/[n]") # text not voice
        selection = ""
        global confirm_input
        if (confirm_input == True):
            selection_filename = "selection"
            selection = speech_to_text(stt_model, selection_filename)
            selection = selection.lower()
            print("\nSelection: " + selection)
        if ( (confirm_input == False) | ("yes" in selection) | ("yeah" in selection) ):
            # Add the user's input to the conversation history
            global conversation_history
            conversation_history += "\nUser: " + user_input

            # Generate a response based on the conversation history
            if (use_gpt == True):
                prompt_tokens = count_tokens(conversation_history)
                if prompt_tokens > 3500:
                    conversation_history = conversation_history.split('\n')
                    conversation_history = conversation_history[0] + '\n' + conversation_history[1] + '\n' + '\n'.join(conversation_history[-300:])
                full_response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[ {"role": "system", "content": conversation_history} ],
                    temperature=0.7,
                    max_tokens=1024,
                    top_p=1,
                    )
                # Extract the response text from the API response
                message = full_response.choices[0].message.content.strip()
            if (use_gpt == False):
                message = "this is a test"

            # Add the response to the conversation history
            conversation_history += "\nChatGPT: " + message
            global convo_save
            if convo_save == True:
                print("saving convo...")
                # save the conversation
                convo_file = open("saved_convo.txt", "w")
                convo_file.write(conversation_history)
                convo_file.close()
            global convo_load, convo_loaded
            if ((convo_load == True) & (convo_loaded == False)):
                convo_loaded = True
                print("loading convo...")
                convo_file = open("saved_convo.txt", "r")
                conversation_history = convo_file.read()
                convo_file.close()

            print()
            print("GLaDOS: ", message)
            if tts_enabled == True:
                tts(message)
    except KeyboardInterrupt:
        print()
        print("\nglados.goodbye()")
        if voicelines == True:
            os.system("mpv sounds/exit_messages/" + random.choice(os.listdir("sounds/exit_messages/")) + " --no-terminal") 
        exit()



def main():
    whisper_model = process_args()
    print("announce.powerup.init()")
    if voicelines == True:
        os.system("mpv sounds/Announcer_wakeup_powerup01.wav --no-terminal")
    if stt_enabled == True:
        print("loading stt_model...")
        stt_model = whisper.load_model(whisper_model + ".en")
        print("stt_model loaded")
    if tts_enabled == True:
        print("loading tts...")
        load_tts()
        print("tts loaded")
    print("announce.powerup.complete()")
    if voicelines == True:
        os.system("mpv sounds/Announcer_wakeup_powerup02.wav --no-terminal") # powerup 
    # i found these to be annoying but you can enable them
    # print("glados.hello()")
    # if voicelines == True:
        # os.system("mpv sounds/welcome_messages/" + random.choice(os.listdir("sounds/welcome_messages/")) + " --no-terminal") 
    if stt_enabled == True:
        while True:
            detect_keyword()
            conversation_loop(stt_model)
    else: 
        while True:
            conversation_loop()



if __name__ == "__main__":
    main()
