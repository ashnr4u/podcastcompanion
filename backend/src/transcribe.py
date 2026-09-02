import json
import os
from pathlib import Path


from faster_whisper import WhisperModel
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
print("BASE_DIR ",BASE_DIR )
AUDIO_DIR = BASE_DIR / "audios"          # changed from VIDEO_DIR
OUTPUT_DIR = BASE_DIR / "data" / "raw_transcripts"

MODEL_SIZE = "base"

AUDIO_EXTENSIONS = {                     # changed from VIDEO_EXTENSIONS
    ".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma"
}


# --------------------------------------------------
# Transcription
# --------------------------------------------------

def transcribe_audio(model, audio_path: Path) -> list[dict]:  # renamed

    print(f"\n[transcribe] {audio_path.name}")

    segments, info = model.transcribe(
        str(audio_path),
        word_timestamps=True
    )

    output = []

    for segment in segments:
        output.append({
            "text": segment.text.strip(),
            "start": segment.start,
            "duration": segment.end - segment.start
        })

    print(
        f"  completed: {info.duration / 60:.1f} min "
        f"({len(output)} segments)"
    )

    return output


# --------------------------------------------------
# Save transcript
# --------------------------------------------------

def write_transcript(
    output_path: Path,
    audio_path: Path,           # renamed from video_path
    segments: list[dict]
):

    data = {
        "file": audio_path.name,   # renamed
        "segments": segments
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"  saved -> {output_path}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    audio_files = []

    for p in AUDIO_DIR.iterdir():
        print("**************************p",p)
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS:
            audio_files.append(p)

    audio_files = sorted(audio_files)

    

    if not audio_files:
        print(f"No audio files found in {AUDIO_DIR}")  # changed
        return

    print(f"Found {len(audio_files)} audio file(s):")

    for audio in audio_files:                # renamed
        print(f"  - {audio.name}")

    print(f"\nLoading Whisper model: {MODEL_SIZE}")

    model = WhisperModel(
        MODEL_SIZE,
        device="cpu",
        compute_type="int8",
        cpu_threads=os.cpu_count() or 4
    )

    for audio_path in audio_files:           # renamed

        output_path = OUTPUT_DIR / f"{audio_path.stem}.json"

        if output_path.exists():
            print(f"\n[skip] {audio_path.name} already processed")
            continue

        segments = transcribe_audio(model, audio_path)   # renamed

        write_transcript(
            output_path,
            audio_path,
            segments
        )

    print("\nDone!")


if __name__ == "__main__":
    main()