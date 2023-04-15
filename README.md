# GLaDOS-GPT
GPT powered personal voice assistant, immitates GLaDOS from Portal.

# Installation requirements
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
./glados.py --help
```
