# ICT Theory Videos — Batch Processing Reference

## Environment (2026-05-12)

| Component | Detail |
|-----------|--------|
| GPU | NVIDIA GeForce GTX 1650 Mobile (4 GB VRAM) |
| CUDA | 12.2 (driver 535.288.01) |
| Python venv | `/home/wek/Documentos/Trading Lucky/venv/` |
| PyTorch | 2.8.0+cu128 |
| Model used | small (244M params, 2 GB VRAM) |

## Video URLs

```
# ICT theory videos — Reafer PIF collection
https://www.youtube.com/watch?v=9Ib9Dm2cnTw    # DELETED — IRRECUPERABLE
https://www.youtube.com/watch?v=tZUshdvodII    # Music/samples — SIN VOZ
https://www.youtube.com/watch?v=7clCk4N9DqA
https://www.youtube.com/watch?v=0PYVJC2FgyA
https://www.youtube.com/watch?v=Ki86EuPGfpQ
https://youtu.be/JSfDrXtKrBE
https://youtu.be/jyc12GURxSk
https://youtu.be/nZJmIltpxpw
https://youtu.be/-o3zOaw1kSY
```

## Replication Command

```bash
# Using Trading Lucky venv (existing CUDA PyTorch)
PYTHONUNBUFFERED=1 /home/wek/Documentos/Trading\ Lucky/venv/bin/python -u \
  /home/wek/.hermes/skills/ai_ops/youtube-transcript/scripts/extract_transcript.py \
  --batch urls_ict.txt \
  --model small
```

## Pitfalls Encountered

1. **pip install PyTorch CUDA OOM'd** — system ran out of RAM downloading 2GB+ CUDA wheels. Solution: reused Trading Lucky venv which already had `torch==2.8.0+cu128`.
2. **torchaudio missing** — VAD failed silently mid-batch because only `faster-whisper` and `yt-dlp` were installed. `torchaudio` must be explicitly installed.
3. **Background process output buffering** — even with `PYTHONUNBUFFERED=1`, the process runner showed empty logs. Monitor with `nvidia-smi` and `ps aux` instead.
4. **GTX 1650 VRAM limit** — 4GB means `small` is the max usable model. `medium` (5GB) would OOM.
