# YouTube Transcript Extractor

GPU-accelerated YouTube transcription pipeline using OpenAI Whisper (faster-whisper)
with CUDA, Voice Activity Detection (VAD), and demucs source separation.

## Features

- GPU-accelerated transcription with faster-whisper (4x faster than openai-whisper)
- Voice Activity Detection (Silero VAD) to isolate speech segments
- Music source separation (demucs htdemucs) for clean vocal extraction
- Batch processing from URL lists
- Automatic audio download + 16kHz mono conversion (yt-dlp + ffmpeg)
- Structured JSON output + human-readable text transcripts
- Caching of downloaded audio and separated vocals
- Comprehensive error handling (deleted/private videos, music-only content)

## Quick Start

```bash
# 1. Install dependencies
pip install faster-whisper yt-dlp torch torchaudio
sudo apt install ffmpeg  # Linux

# Optional: music separation
pip install demucs

# 2. Check environment
python scripts/extract_transcript.py --check-env

# 3. Transcribe a video
python scripts/extract_transcript.py "https://youtube.com/watch?v=dQw4w9WgXcQ"

# 4. For music videos (avoids hallucination)
python scripts/extract_transcript.py "URL" --no-vad --demucs
```

## Usage

```bash
# Basic
python scripts/extract_transcript.py <youtube_url_or_id>

# Options
--model MODEL        Whisper model: tiny|base|small|medium|large-v2|large-v3|turbo
--no-vad             Disable VAD (use for music-heavy content)
--demucs             Separate vocals from music before transcribing
--language LANG      Language hint (en, es, ja, etc.)
--batch FILE         Process multiple URLs from file (one per line)
--check-env          Verify all dependencies and exit
--json-only          Output JSON to stdout instead of writing files
--vad-threshold 0.5  VAD sensitivity (0-1, lower = more sensitive)
```

## Management Commands

```bash
python scripts/action.py check-env       # Environment status
python scripts/action.py list            # List all transcripts
python scripts/action.py search "query"  # Search transcripts
python scripts/action.py stats           # Usage statistics
python scripts/action.py diagnose ID     # Diagnose a video
```

## Model Selection

| Model | Params | VRAM | Speed | Use Case |
|-------|--------|------|-------|----------|
| tiny | 39M | 1 GB | 10x | Quick drafts |
| base | 74M | 1 GB | 7x | Casual use |
| small | 244M | 2 GB | 4x | Good quality |
| medium | 769M | 5 GB | 2x | **Default — best balance** |
| turbo | 809M | 6 GB | 8x | Low-VRAM, fast |
| large-v3 | 1.5B | 10 GB | 1x | Maximum accuracy |

## When Things Go Wrong

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| "Video unavailable" | Deleted or private | Unrecoverable — check archive.org |
| Repeated/looping text | Music hallucination | Add `--no-vad --demucs` |
| CUDA out of memory | Model too large | Use `--model turbo` or `small` |
| Empty output | No speech in video | Check `speech_ratio` in JSON |
| ffmpeg not found | Missing system dep | `sudo apt install ffmpeg` |

## Architecture

```
URL → yt-dlp (download WAV 16kHz mono)
    → [demucs (separate vocals)]
    → [Silero VAD (detect speech)]
    → faster-whisper (transcribe on GPU)
    → JSON + TXT output
```

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA (optional — CPU fallback works)
- ffmpeg (system package)
- 5 GB VRAM for medium model, 10 GB for large-v3

## License

MIT
