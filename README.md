# WARNING: This is a dev branch!!!
this means that the code here is likely broken, only use if you know what youre doing

# GLaDOS-GPT Features
- wake word 
- highly accurate speech to text (with bigger whisper models)
- GLaDOS voice immitation
- sense of humor, will make mean jokes just like GLaDOS
- real voicelines 
- expandability - this project is open source so you can add your own modules!

# Video example
[![Oops! There should have been a video example.](thumbnail.jpg)](https://streamable.com/2ry356)

# Installation
python modules (i might be missing a few, create an issue if i am):
```
pip3 install openai torch openai-whisper rhasspy-silence pvporcupine
```
global variables (like in .bashrc):
```
export OPENAI_API_KEY="keyhere"
export PICOVOICE_KEY="keyhere"
```
get the keys from 
- https://platform.openai.com/account/api-keys
- https://console.picovoice.ai/

download
```
git clone https://github.com/KawaiiKraken/GLaDOS-GPT
```
run
```
cd GLaDOS-GPT
./glados.py --help # to get options
./glados.py
```

# Upcoming features (if there is interest)
- persistent conversations/long term memory
- smart home integration
- GUI
- automating standart computer tasks 
- more configuration
- more?


# How to contribute 
Thank you for considering contributing to our project! Here are some ways you can get involved:

- Give us a star on GitHub to show your support.
- Add problems or feature requests to our issue tracker.
- Fork our project and make your own modifications, and submit pull requests with your changes.
- Spread the word about our project on social media and other platforms.
- Help us improve our documentation or fix bugs.
- Donate to support our project, if you are able to.

We appreciate any contributions, big or small, and look forward to collaborating with you!
