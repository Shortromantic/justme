import openai
import os 
import logging
from pydub import AudioSegment
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def oga_2_mp3(filename):
    """
    Converts an OGA audio file to MP3 format using pydub.

    Args:
        filename (str): The base filename without extension.

    Returns:
        str: The path to the converted MP3 file.
    """
    input_file = f"{filename}.oga"
    output_file = f"{filename}.mp3"

    try:
        # Load .oga file
        audio = AudioSegment.from_ogg(input_file)

        # Export as .mp3
        audio.export(output_file, format="mp3")
        logger.info(f"Converted {input_file} to {output_file}")
        
        return output_file
    except Exception as e:
        logger.error(f"Failed to convert {input_file} to MP3: {e}")
        raise

def oga_2_mp3_2_text(filename):
    """
    Converts an OGA audio file to MP3 and transcribes it to text using OpenAI's Whisper API.

    Args:
        filename (str): The base filename without extension.

    Returns:
        str: The transcribed text from the audio file.
    """
    mp3_file_path = oga_2_mp3(filename)
    openai.api_key = OPENAI_API_KEY

    transcript = None

    try:
        with open(mp3_file_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file, language="en")
            logger.info(f"Transcription successful for {mp3_file_path}")
    except Exception as e:
        logger.error(f"Transcription failed for {mp3_file_path}: {e}")
        return "Sorry, there was an error processing the audio."

    # Delete audio files if the transcription was successful
    if transcript:
        os.remove(mp3_file_path)
        os.remove(f"{filename}.oga")
        logger.info("Audio files deleted after successful transcription.")
    else:
        logger.warning(f"Audio files not deleted because transcription failed for {filename}.")

    return transcript['text'] if transcript else ""
