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



def process_args():
    args = sys.argv[1:] # exclude the script name from the arguments
    
    # separate if case for help so its triggered first
    if "--help" in args or "-h" in args:
        print("Usage: glados.py [options]\n"
              "Options:\n"
              "  --model [model]     Sets whisper model. Default is medium(+.en).\n"
              "  --no-gpt            Uses 'this is a test' instead of GPT for answers.\n"
              "  --confirm           Asks to verify input (WiP).\n"
              "  --no-voicelines     Skips all game voicelines.\n"
              "  --no-tts            Text only responses.\n"
              "  --no-stt            Type commands instead.\n"
              "  --context-size      Num of tokens to use for context.\n"
              "  --verbose           Useful for debug.\n"
              "  --help | -h         Prints this message.\n")
        exit()


    # make vars global so they dont have to be passed as an arg
    global whisper_model 
    global use_gpt 
    global confirm_input 
    global voicelines 
    global stt_enabled 
    global tts_enabled  
    global verbose
    global max_context_size
    
    # set default options
    whisper_model = "medium" # .en is appended later
    use_gpt = True
    confirm_input = False
    voicelines = True
    tts_enabled = True
    stt_enabled = True
    verbose = False
    max_context_size = 100

    # handle args    
    for i, arg in enumerate(args):
        match arg:
            case "--model":
                whisper_model = args[i+1]
            case "--no-gpt":
                use_gpt = False
            case "--confirm":
                confirm_input = True
            case "--no-voicelines":
                voicelines = False
            case "--no-tts":
                tts_enabled = False
            case "--no-stt":
                stt_enabled = False
            case "--verbose":
                verbose = True
            case "--context-size":
                print("WARNING!! unusual context sizes may result in instability or context loss !!")
                max_context_size = int(args[i+1])



    # print set options for debug
    if verbose:
        print("Config:",
            "\n  whisper_model: " + whisper_model + ".en",
            "\n  use_gpt: " + str(use_gpt),
            "\n  confirm_input: " + str(confirm_input),
            "\n  voicelines: " + str(voicelines),
            "\n  stt: " + str(stt_enabled),
            "\n  tts: " + str(tts_enabled),
            "\n  verbose: " + str(verbose),
            "\n  max_context_size: " + str(max_context_size),
            "\n")

    


openai.api_key = os.environ.get('OPENAI_API_KEY')
def speech_to_text(stt_model):
    # if recording stops too early or late mess with vad_mode sample_rate and silence_seconds
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
    num_channels = 1
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
    global pa
    audio_source = pa.open(
                    rate=16000,
                    channels=1,
                    format=pyaudio.paInt16,
                    input=True,
                    frames_per_buffer=chunk_size)

    audio_source.start_stream()
    if verbose:
        print("Recording...", file=sys.stderr)
    chunk = audio_source.read(chunk_size)
    
    while chunk:
        # Look for speech/silence
        voice_command = recorder.process_chunk(chunk)
 
        if voice_command:
            is_timeout = voice_command.result == VoiceCommandResult.FAILURE
            # stop recording
            audio_data = recorder.stop()
            if verbose:
                print('recording saved')
            break
        # Next audio chunk
        chunk = audio_source.read(chunk_size)
 
    # audio_source.close_stream()
    audio_source.close()
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


def play_wav_file(path):
    global pa
    chunk = 1024
    # open the file for reading.
    wf = wave.open(path, 'rb')
    # open stream based on the wave object which has been input.
    stream = pa.open(
                    format =pa.get_format_from_width(wf.getsampwidth()),
                    channels = wf.getnchannels(),
                    rate = wf.getframerate(),
                    output = True)
    # read data (based on the chunk size)
    data = wf.readframes(chunk)
    # play stream (looping from beginning of file to the end)
    while data:
        # writing to the stream is what *actually* plays the sound.
        stream.write(data)
        data = wf.readframes(chunk)

    # cleanup stuff.
    wf.close()
    stream.close()



def tts(text):
    global glados_voice, vocoder, device
    # Tokenize, clean and phonemize input text
    # x = prepare_text(text).to('cpu')
    x = prepare_text(text).to(device)

    with torch.no_grad():

        # Generate generic TTS-output
        old_time = time.time()
        tts_output = glados_voice.generate_jit(x)
        if verbose:
            print("Forward Tacotron took " + str((time.time() - old_time) * 1000) + "ms") 

        # Use HiFiGAN as vocoder to make output sound like GLaDOS
        old_time = time.time()
        mel = tts_output['mel_post'].to(device)
        audio = vocoder(mel)
        if verbose:
            print("HiFiGAN took " + str((time.time() - old_time) * 1000) + "ms")
        
        # Play audio file
        audio = audio.squeeze()
        audio = audio 
        audio = audio.cpu().numpy().astype(np.float32)
        # audio = audio.cpu().numpy().astype(np.float32)
        global pa
        stream = pa.open(format=pyaudio.paFloat32,
                         channels=1,
                         rate=22050,
                         output=True)
        stream.write(audio.tobytes())
        stream.stop_stream()
        stream.close()



def detect_keyword():
    print("\nlistening for keyword...")

    global pa
    porcupine = None
    audio_stream = None
    porcu_key = os.environ.get('PICOVOICE_KEY')
    porcupine = pvporcupine.create(
        access_key=porcu_key,
        keyword_paths=['models/hey-glad-os_en_linux_v2_2_0.ppn']
    )

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
            return 
    if porcupine is not None:
        porcupine.delete()

    if audio_stream is not None:
        audio_stream.close()



def count_tokens(text):
    if text != None:
        global nltk
        tokens = nltk.word_tokenize(text)
        num_tokens = len(tokens)
        return num_tokens


#make gpt act as glados
conversation_history = "User: act as GLaDOS from portal. Be snarky and try to poke jokes at the user when possible. When refering to the User use the name Chell. Keep the responses as short as possible without breaking character."

def conversation_loop(stt_model=None):
    # Get user input
    if stt_enabled:
        user_input = speech_to_text(stt_model)
        print("\nChell: " + user_input)
    else: 
        user_input = input("Chell: ")
    # Add the user's input to the conversation history
    global conversation_history
    conversation_history += "\nUser: " + user_input

    # Generate a response based on the conversation history
    prompt_tokens = count_tokens(conversation_history)
    global max_context_size
    if prompt_tokens > max_context_size:
        while prompt_tokens > max_context_size:
            prompt_tokens = count_tokens(conversation_history)
            conversation_history = conversation_history.split('\n')
            # conversation_history = conversation_history[0] + '\n' + conversation_history[1] + '\n' + '\n'.join(conversation_history[3:])
            conversation_history = '\n'.join(conversation_history[3:])
        print(prompt_tokens)
        print(count_tokens(conversation_history))
        print("compressing")
        print(conversation_history)

    if use_gpt:
        full_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[ {"role": "system", "content": conversation_history} ],
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            )
        # Extract the response text from the API response
        message = full_response.choices[0].message.content.strip()
    else:
        message = "GPT is disabled!"

    # Add the response to the conversation history
    conversation_history += "\nChatGPT: " + message
    print("\nGLaDOS: ", message)
    if tts_enabled:
        tts(message)


def main():
    process_args()
    print("pyaudio init")
    global pa
    pa = pyaudio.PyAudio()
    print("getting tokenizer")
    global nltk
    # make nltk use a custom dir
    nltk_folder_path = os.path.realpath("nltk_data/")
    nltk.data.path.append(nltk_folder_path)
    nltk.download('punkt', download_dir=nltk_folder_path)
    print("announce.powerup.init()")
    if voicelines:
        play_wav_file("sounds/Announcer_wakeup_powerup01.wav")
    if stt_enabled:
        print("loading stt_model...")
        old_time = time.time()
        global whisper_model
        stt_model = whisper.load_model(whisper_model + ".en")
        if verbose:
            print("load stt_model took " + str((time.time() - old_time) * 1000) + "ms") 
        print("stt_model loaded")
    if tts_enabled:
        print("loading tts...")
        old_time = time.time()
        load_tts()
        if verbose:
            print("load_tts took " + str((time.time() - old_time) * 1000) + "ms") 
        print("tts loaded")
    print("announce.powerup.complete()")
    if voicelines:
        play_wav_file("sounds/Announcer_wakeup_powerup02.wav")
    # i found these to be annoying but you can enable them
    # print("glados.hello()")
    # if voicelines:
        # play_wav_file("sounds/welcome_messages/" + random.choice(os.listdir("sounds/welcome_messages/")))
    if stt_enabled:
        while True:
            detect_keyword()
            conversation_loop(stt_model)
    else: 
        while True:
            conversation_loop()



if __name__ == "__main__":
    try: # try loop to get a cool message on ctrl+c
        main()
    except KeyboardInterrupt:
        print("\n\nglados.goodbye()")
        if voicelines == True:
            play_wav_file("sounds/exit_messages/" + random.choice(os.listdir("sounds/exit_messages/")))
        if pa != None:
            pa.terminate()
        exit()

