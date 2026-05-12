---
name: youtube-transcript
description: "GPU-accelerated YouTube transcription with Whisper, VAD, and demucs source separation for production-grade transcripts."
platforms: [linux]
category: ai_ops
---

# YouTube Transcript Extractor — GPU Pipeline

Production-grade tool to extract high-quality transcripts from YouTube videos using
OpenAI Whisper (via faster-whisper) with CUDA acceleration, Voice Activity Detection
(VAD), and optional demucs source separation for music-heavy content.

## When to Use

- User shares a YouTube URL and needs a transcript
- Processing music videos or content with background music (use `--demucs`)
- Bulk transcription from a list of URLs (`--batch`)
- User asks "what's in this video?" or "summarize this YouTube"
- Need speaker-ready transcripts with timestamps

## Quick Start

```bash
# Single video
python scripts/extract_transcript.py "https://youtube.com/watch?v=VIDEO_ID"

# Music-heavy video (avoids hallucination)
python scripts/extract_transcript.py "URL" --no-vad --demucs

# Batch processing
python scripts/extract_transcript.py --batch urls.txt --model medium

# Check environment
python scripts/extract_transcript.py --check-env
```

## Prerequisites

```bash
# Core (all four required — torchaudio enables VAD)
pip install faster-whisper yt-dlp torch torchaudio

# Optional: source separation
pip install demucs

# System
sudo apt install ffmpeg  # Linux
brew install ffmpeg       # macOS
```

**Smart setup — reuse existing venv with CUDA PyTorch:** If reinstalling PyTorch OOMs or takes too long, scan the user's projects for an existing venv with CUDA PyTorch (`find ~/Documentos ~/Escritorio -path '*/venv/bin/python' -exec {} -c "import torch; print(torch.cuda.is_available())" \;`). Install only missing deps (`faster-whisper yt-dlp torchaudio`) into that venv. Avoids re-downloading 2GB+ CUDA PyTorch wheels.

## Commands Reference

### Model Selection (VRAM constraints matter)
| Model | Params | VRAM | Speed | Best For |
|-------|--------|------|-------|----------|
| tiny | 39M | 1 GB | 10x | Quick drafts, CPU fallback |
| base | 74M | 1 GB | 7x | Casual use |
| small | 244M | 2 GB | 4x | Good quality, low-VRAM GPUs (GTX 1650) |
| medium | 769M | 5 GB | 2x | **Best balance (default)** |
| turbo | 809M | 6 GB | 8x | Speed-optimized |
| large-v3 | 1.5B | 10 GB | 1x | Maximum accuracy |

### VAD + Demucs Decision Matrix
| Content Type | VAD | Demucs | Flag |
|-------------|-----|--------|------|
| Podcast / Lecture | ON | OFF | (default) |
| Music video (vocals+instrumental) | OFF | ON | `--no-vad --demucs` |
| Electronic music / EDM | OFF | ON | `--no-vad --demucs` |
| Pure instrumental (no voice) | OFF | OFF | Skip — no transcript possible |
| Mixed content | ON | ON | Slowest, best quality |

### extract_transcript.py — Main Pipeline

| Flag | Description | When to Use |
|------|-------------|-------------|
| `--model medium` | Whisper model (default: medium) | medium = best balance |
| `--model large-v3` | Highest accuracy | Critical/legal content |
| `--model turbo` | Speed-optimized | Low-VRAM GPUs |
| `--no-vad` | Disable Voice Activity Detection | Music videos, EDM |
| `--demucs` | Separate vocals from music | Any video with background music |
| `--language es` | Language hint | Non-English content |
| `--vad-threshold 0.3` | Lower = more sensitive VAD | Quiet speakers |
| `--batch urls.txt` | Process multiple videos | Bulk transcription |
| `--check-env` | Verify all dependencies | First run |
| `--json-only` | Output JSON to stdout | Pipe to other tools |

### action.py — Management Actions

```bash
python scripts/action.py check-env       # Dependency check
python scripts/action.py list            # List all transcripts
python scripts/action.py search "query"  # Full-text search
python scripts/action.py stats           # Usage statistics
python scripts/action.py diagnose ID     # Diagnose specific video
```

## Workflow

1. **Verify environment**: `python scripts/extract_transcript.py --check-env`
2. **Process video**: `python scripts/extract_transcript.py "URL"`
3. **Check result**: Transcript saved to `transcripts/VIDEO_ID.txt` + `.json`
4. **Iterate if needed**: Use `--no-vad --demucs` for music; switch model for accuracy

## Failure Modes & Recovery

| Error | Cause | Solution |
|-------|-------|----------|
| "Video unavailable" | Deleted/private | IRRECUPERABLE — buscar copia local |
| Hallucination (loops) | Music without VAD | Add `--no-vad --demucs` |
| OOM error | GPU VRAM exceeded | Use `--model small` or `turbo` |
| Empty transcript | No speech content | Check speech_ratio in JSON output |
| ffmpeg not found | System dependency | `sudo apt install ffmpeg` |
| torchaudio not found (VAD fails) | Missing dep after partial venv setup | `pip install torchaudio` — VAD import is lazy, `--check-env` catches it but batch mode may surface it mid-run |
| Background process shows no output | Python stdout buffering + non-TTY pipe | Use `PYTHONUNBUFFERED=1` and `python -u`, or monitor via `nvidia-smi` + `ps aux`; use `--json-only` to stdout for real-time feedback in foreground |
| pip install OOM (PyTorch CUDA) | 2GB+ wheel download exhausts RAM | Reuse existing CUDA PyTorch venv from user's projects instead of reinstalling |

## Output Files

- `transcripts/VIDEO_ID.txt` — Human-readable transcript with timestamps
- `transcripts/VIDEO_ID.json` — Structured data (segments, metadata, timings)
- `transcripts/batch_summary.json` — Batch processing results
- `cache/audio/VIDEO_ID.wav` — Cached audio (safe to delete)

## GPU Troubleshooting

### CUDA not detected
```bash
# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA
pip uninstall torch torchaudio -y
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Out of Memory
- Use `--model turbo` (809M params, 8x realtime, ~6GB VRAM)
- Disable demucs
- Set `compute_type="int8"` for lower precision

### Slow transcription
- Verify CUDA is actually used: `nvidia-smi` during transcription
- Try `--model turbo` for 8x realtime speed
- Ensure audio is 16kHz mono (handled automatically by yt-dlp)

## Integration with Hermes

From any conversation:
```bash
# Fetch and summarize
python scripts/extract_transcript.py "URL" --json-only | python scripts/summarize.py
```
