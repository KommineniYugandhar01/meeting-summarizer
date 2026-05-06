import os
import torch
import whisper
from gtts import gTTS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


os.environ["PATH"] += os.pathsep + r'C:\ffmpeg'
AUDIO_FILE = "meeting_audio.wav"
#step 1:
print("🔊 Step 1: Generating meeting audio file...")
meeting_text = """
Hello everyone. Thank you for joining today's project sync. 
Our main topic is the launch of the new AI platform. 
Specifically, Rahul needs to finish the database migration by Friday. 
Sneha will handle the front-end testing. 
We must have the final report ready before the deadline next Tuesday. 
Does anyone have questions? Okay, meeting adjourned.
"""
tts = gTTS(text=meeting_text, lang='en')
tts.save(AUDIO_FILE)
print(f"✅ Created {AUDIO_FILE}")

#step 2
print("\n👂 Step 2: Transcribing audio with Whisper AI...")
whisper_model = whisper.load_model("base")
# result is created here for the summarizer to use later
result = whisper_model.transcribe(AUDIO_FILE, fp16=False)
print("✅ Transcription Complete.")
print(f"Captured Text: {result['text'][:100]}...")

#STEP 3:
print("\n🧠 Step 3: Summarizing with Local BART Model...")
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Prepare the input for the model
input_text = "summarize: " + result['text']
inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)

# Generate Summary IDs
summary_ids = model.generate(
    inputs["input_ids"], 
    max_length=150, 
    min_length=40, 
    length_penalty=2.0, 
    num_beams=4, 
    early_stopping=True
)
final_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("\n" + "="*45)
print("✨ FINAL PROACTIVE MEETING SUMMARY ✨")
print("="*45)
# Remove the 'summarize:' prefix if the model added it
clean_summary = final_summary.replace("summarize:", "").strip()
print(clean_summary)
print("="*45)