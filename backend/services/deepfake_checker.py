import os
import sys
from typing import Set

try:
    import torch
    from transformers import pipeline
except ImportError:
    print("="*80)
    print("ERROR: Missing critical libraries.")
    print("Please install all required dependencies first:")
    print("pip install torch transformers")
    print("="*80)
    sys.exit(1)

# --- Configuration ---
AUDIO_FORMATS: Set[str] = {'.mp3', '.wav', '.m4a', '.flac', '.ogg'}
DEVICE = 0 if torch.cuda.is_available() else -1  # 0 for CUDA, -1 for CPU
AUDIO_MODEL_ID = "mo-thecreator/Deepfake-audio-detection"

audio_pipeline_instance = None

def get_audio_pipeline():
    """Loads the audio pipeline into memory (if not already loaded)."""
    global audio_pipeline_instance
    if audio_pipeline_instance is None:
        try:
            print(f"Loading audio model '{AUDIO_MODEL_ID}' from Hugging Face Hub...")
            audio_pipeline_instance = pipeline(
                "audio-classification",
                model=AUDIO_MODEL_ID,
                device=DEVICE
            )
            print("Audio detection pipeline loaded successfully.")
        except Exception as e:
            print(f"Error loading audio pipeline: {e}")
            print("Please ensure the model ID is correct.")
            sys.exit(1)
    return audio_pipeline_instance

def detect_audio_deepfake(file_path: str) -> bool:
    """
    Runs a pretrained audio deepfake detection model from the HF Hub.
    """
    print(f"Analyzing audio file: {os.path.basename(file_path)}")
    try:
        detector = get_audio_pipeline()
    except Exception as e:
        print(f"Failed to load audio pipeline: {e}")
        return False  # Fail safe
    try:
        results = detector(file_path)
        best_result = max(results, key=lambda x: x['score'])
        top_label = best_result['label'].lower()
        top_score = best_result['score']
        print(f"...Audio pipeline result: '{top_label}' with score {top_score:.4f}")
        is_fake = top_label in ['spoof', 'fake']
        return is_fake
    except Exception as e:
        print(f"Error during audio processing/inference: {e}")
        return False

def is_audio_deepfake(file_path: str) -> bool:
    """
    Checks if a given audio file is a deepfake.
    Args:
        file_path: The absolute or relative path to the audio file.
    Returns:
        True if the file is classified as a deepfake, False otherwise.
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at path: {file_path}")
    ext = os.path.splitext(file_path)[1].lower()
    if ext in AUDIO_FORMATS:
        return detect_audio_deepfake(file_path)
    else:
        raise ValueError(
            f"Unsupported file format: {ext}. Supported types: {AUDIO_FORMATS}"
        )

