import os
import gradio as gr
import whisper
from deep_translator import GoogleTranslator
from gtts import gTTS

# Load Whisper Model
model = whisper.load_model("base")

# Define Language Codes
LANGUAGE_CODES = {
    "Urdu": "ur", "Hindi": "hi", "French": "fr", "Spanish": "es",
    "German": "de", "Chinese": "zh-TW", "Arabic": "ar", "Russian": "ru"
}

# Function to Split Text into 5000-Character Chunks
def split_text(text, max_length=5000):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

# Function to Translate Long Text
def translate_large_text(text, target_lang):
    chunks = split_text(text, 5000)  # Split into 5000-character chunks
    translated_chunks = [GoogleTranslator(source="en", target=target_lang).translate(chunk) for chunk in chunks]
    return " ".join(translated_chunks)

# Function to Generate Audio from Text
def generate_audio(text, language_code):
    tts = gTTS(text=text, lang=language_code, slow=False)
    audio_file = "translated_audio.mp3"
    tts.save(audio_file)
    return audio_file

# Process Audio Function
def process_audio(audio_file, target_language):
    if not audio_file:
        return "⚠️ No audio file provided.", "", None

    # Transcribe with Whisper
    result = model.transcribe(audio_file)
    transcription = result.get("text", "").strip()

    if not transcription:
        return "⚠️ No speech detected in the audio.", "", None

    # Translate the text with handling for large text
    target_code = LANGUAGE_CODES.get(target_language, "ur")
    translated_text = translate_large_text(transcription, target_code)

    # Generate audio from translated text
    translated_audio = generate_audio(translated_text, target_code)

    return transcription, translated_text, translated_audio

# Gradio Interface
def gradio_interface(audio_file, target_language):
    if audio_file is None:
        return "⚠️ Please upload an audio file.", "", None

    transcription, translated_text, translated_audio = process_audio(audio_file, target_language)
    return transcription, translated_text, translated_audio

# Gradio UI
input_components = [
    gr.Audio(type="filepath", label="📂 Upload an MP3/WAV file"),
    gr.Dropdown(choices=list(LANGUAGE_CODES.keys()), label="🌍 Select Translation Language")
]

output_components = [
    gr.Textbox(label="📝 Transcription (English)"),
    gr.Textbox(label="🌎 Translation"),
    gr.Audio(label="🎧 Translated Audio", type="filepath")
]

interface = gr.Interface(
    fn=gradio_interface,
    inputs=input_components,
    outputs=output_components,
    title="🎙️ Speech to Text & Translator 🌍",
    description="Upload an audio file, select a language, and get transcription + translation + translated audio."
)

# Launch Gradio App
interface.launch(share=True)
