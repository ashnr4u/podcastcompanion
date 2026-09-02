import json
import os
from pathlib import Path

from faster_whisper import WhisperModel


# Prevent unnecessary Hugging Face symlink warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

AUDIO_DIR = BASE_DIR / "audios"
OUTPUT_DIR = BASE_DIR / "data" / "raw_transcripts"


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_SIZE = "base"

AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".flac",
    ".m4a",
    ".aac",
    ".ogg",
    ".opus",
    ".wma",
}


# --------------------------------------------------
# Transcription
# --------------------------------------------------

def transcribe_audio(
    model,
    audio_path: Path
) -> list[dict]:

    print(f"\n[transcribe] {audio_path.name}")

    segments, info = model.transcribe(
        str(audio_path)
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
    audio_path: Path,
    segments: list[dict]
):

    data = {
        "file": audio_path.name,
        "segments": segments
    }

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"  saved -> {output_path}")


# --------------------------------------------------
# Transcribe all new audio
# --------------------------------------------------

def transcribe_all():

    print("\nStarting transcription...\n")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not AUDIO_DIR.exists():

        print(
            f"Audio directory not found: {AUDIO_DIR}"
        )

        return

    audio_files = [
        p
        for p in AUDIO_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in AUDIO_EXTENSIONS
    ]

    audio_files = sorted(audio_files)

    if not audio_files:

        print(
            f"No audio files found in {AUDIO_DIR}"
        )

        return

    print(
        f"Found {len(audio_files)} audio file(s):"
    )

    for audio in audio_files:
        print(f"  - {audio.name}")


    # --------------------------------------------------
    # Check which files actually need transcription
    # --------------------------------------------------

    files_to_process = []

    for audio_path in audio_files:

        output_path = (
            OUTPUT_DIR
            / f"{audio_path.stem}.json"
        )

        if output_path.exists():

            print(
                f"\n[skip] "
                f"{audio_path.name} already processed"
            )

        else:

            files_to_process.append(
                audio_path
            )


    # No new files
    if not files_to_process:

        print("\nNo new audio files to transcribe.")

        return


    # --------------------------------------------------
    # Load Whisper only when needed
    # --------------------------------------------------

    print(
        f"\nLoading Whisper model: {MODEL_SIZE}"
    )

    model = WhisperModel(
        MODEL_SIZE,
        device="cpu",
        compute_type="int8",
        cpu_threads=os.cpu_count() or 4
    )


    # --------------------------------------------------
    # Process new audio
    # --------------------------------------------------

    for audio_path in files_to_process:

        output_path = (
            OUTPUT_DIR
            / f"{audio_path.stem}.json"
        )

        segments = transcribe_audio(
            model,
            audio_path
        )

        write_transcript(
            output_path,
            audio_path,
            segments
        )


    print("\nTranscription complete.")


# --------------------------------------------------
# Standalone execution
# --------------------------------------------------

if __name__ == "__main__":
    transcribe_all()