#!/usr/bin/env python3
"""
action.py — Hermes Agent Integration Actions
=============================================

Actions callable from SKILL.md workflows or directly by Hermes.
Each function returns structured JSON for tool chaining.

Usage (from Hermes):
  python scripts/action.py check-env
  python scripts/action.py list-transcripts
  python scripts/action.py search "bitcoin"
  python scripts/action.py stats
  python scripts/action.py diagnose VIDEO_ID
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

SKILL_DIR = Path(__file__).resolve().parent.parent
TRANSCRIPTS_DIR = SKILL_DIR / "transcripts"
CACHE_DIR = SKILL_DIR / "cache"


def action_check_env() -> dict:
    """Check all dependencies and return environment status."""
    import subprocess

    checks = {}

    # Python version
    checks["python"] = {
        "version": sys.version.split()[0],
        "path": sys.executable,
    }

    # CUDA / PyTorch
    try:
        import torch
        checks["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            checks["torch"]["device"] = torch.cuda.get_device_name(0)
            checks["torch"]["vram_mb"] = (
                torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            )
    except ImportError:
        checks["torch"] = {"error": "not installed"}

    # faster-whisper
    try:
        import faster_whisper
        checks["faster_whisper"] = {"status": "installed"}
    except ImportError:
        checks["faster_whisper"] = {"error": "not installed — pip install faster-whisper"}

    # yt-dlp
    try:
        result = subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True)
        checks["yt_dlp"] = {"version": result.stdout.strip()}
    except FileNotFoundError:
        checks["yt_dlp"] = {"error": "not installed — pip install yt-dlp"}

    # ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True)
        checks["ffmpeg"] = {"status": "available"}
    except FileNotFoundError:
        checks["ffmpeg"] = {"error": "not installed — sudo apt install ffmpeg"}

    # demucs
    try:
        subprocess.run(
            ["python", "-c", "from demucs import separate"],
            capture_output=True, timeout=10,
        )
        checks["demucs"] = {"status": "available"}
    except (subprocess.CalledProcessError, FileNotFoundError):
        checks["demucs"] = {"error": "not installed — pip install demucs"}

    # VAD
    try:
        import torchaudio
        checks["silero_vad"] = {"status": "available (via torch.hub)"}
    except ImportError:
        checks["silero_vad"] = {"error": "torchaudio not installed — pip install torchaudio"}

    # Ready?
    checks["ready"] = all(
        "error" not in checks.get(k, {})
        for k in ["torch", "faster_whisper", "yt_dlp", "ffmpeg"]
    )

    return checks


def action_list_transcripts(limit: int = 50) -> dict:
    """List all transcripts in the transcripts directory."""
    if not TRANSCRIPTS_DIR.exists():
        return {"count": 0, "transcripts": [], "message": "No transcripts directory"}

    txt_files = sorted(TRANSCRIPTS_DIR.glob("*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    json_files = sorted(TRANSCRIPTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    transcripts = []
    for tf in txt_files[:limit]:
        video_id = tf.stem
        json_path = TRANSCRIPTS_DIR / f"{video_id}.json"

        info = {
            "video_id": video_id,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "txt_path": str(tf),
            "size_kb": round(tf.stat().st_size / 1024, 1),
            "modified": datetime.fromtimestamp(tf.stat().st_mtime).isoformat(),
        }

        # Enrich from JSON
        if json_path.exists():
            try:
                data = json.loads(json_path.read_text())
                info["duration"] = data.get("duration_formatted", "?")
                info["language"] = data.get("language", "?")
                info["model"] = data.get("model", "?")
                info["segments"] = data.get("segment_count", 0)
            except (json.JSONDecodeError, KeyError):
                pass

        transcripts.append(info)

    return {
        "count": len(transcripts),
        "total_json": len(json_files),
        "total_txt": len(txt_files),
        "transcripts": transcripts,
    }


def action_search(query: str, limit: int = 20) -> dict:
    """Full-text search across all transcripts."""
    if not TRANSCRIPTS_DIR.exists():
        return {"query": query, "matches": [], "count": 0}

    query_lower = query.lower()
    matches = []

    for txt_file in sorted(TRANSCRIPTS_DIR.glob("*.txt"),
                           key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            content = txt_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        if query_lower in content.lower():
            # Extract context lines
            lines = content.split("\n")
            hit_lines = []
            for i, line in enumerate(lines):
                if query_lower in line.lower():
                    context_start = max(0, i - 1)
                    context_end = min(len(lines), i + 2)
                    hit_lines.append({
                        "line_num": i + 1,
                        "context": "\n".join(lines[context_start:context_end]),
                    })
                    if len(hit_lines) >= 5:
                        break

            matches.append({
                "video_id": txt_file.stem,
                "url": f"https://www.youtube.com/watch?v={txt_file.stem}",
                "hits": len(hit_lines),
                "samples": hit_lines[:3],
            })

        if len(matches) >= limit:
            break

    return {
        "query": query,
        "matches": matches,
        "count": len(matches),
    }


def action_stats() -> dict:
    """Compute statistics across all transcripts."""
    if not TRANSCRIPTS_DIR.exists():
        return {"error": "No transcripts directory"}

    json_files = list(TRANSCRIPTS_DIR.glob("*.json"))

    models_used = {}
    languages = {}
    total_duration = 0.0
    total_segments = 0
    total_processing_time = 0.0

    for jf in json_files:
        try:
            data = json.loads(jf.read_text())
        except (json.JSONDecodeError, KeyError):
            continue

        model = data.get("model", "unknown")
        models_used[model] = models_used.get(model, 0) + 1

        lang = data.get("language", "unknown")
        languages[lang] = languages.get(lang, 0) + 1

        total_duration += data.get("duration_seconds", 0)
        total_segments += data.get("segment_count", 0)
        total_processing_time += data.get("elapsed_s", 0)

    return {
        "total_transcripts": len(json_files),
        "total_duration_hours": round(total_duration / 3600, 2),
        "total_duration_formatted": f"{int(total_duration // 3600)}h {int((total_duration % 3600) // 60)}m",
        "total_segments": total_segments,
        "total_processing_time_s": round(total_processing_time, 1),
        "average_rtf": round(total_processing_time / total_duration, 2) if total_duration > 0 else None,
        "models_used": models_used,
        "languages": languages,
        "cache_size_mb": _get_cache_size(),
    }


def action_diagnose(video_id: str) -> dict:
    """Diagnose a specific video: check if transcript exists, verify audio, etc."""
    result = {
        "video_id": video_id,
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "checks": {},
    }

    # Check transcripts
    txt_path = TRANSCRIPTS_DIR / f"{video_id}.txt"
    json_path = TRANSCRIPTS_DIR / f"{video_id}.json"

    result["checks"]["transcript_txt"] = txt_path.exists()
    result["checks"]["transcript_json"] = json_path.exists()

    if json_path.exists():
        data = json.loads(json_path.read_text())
        result["metadata"] = {
            "duration": data.get("duration_formatted"),
            "language": data.get("language"),
            "model": data.get("model"),
            "segments": data.get("segment_count"),
            "speech_ratio": data.get("speech_ratio"),
        }

    # Check audio cache
    audio_path = CACHE_DIR / "audio" / f"{video_id}.wav"
    vocals_path = CACHE_DIR / "audio" / "htdemucs" / video_id / "vocals.wav"

    result["checks"]["audio_cached"] = audio_path.exists()
    result["checks"]["vocals_cached"] = vocals_path.exists()

    if audio_path.exists():
        result["audio"] = {
            "size_mb": round(audio_path.stat().st_size / (1024 * 1024), 1),
        }

    # Quick health check
    all_ok = all(result["checks"].values())
    result["status"] = "healthy" if all_ok else "incomplete"

    if not result["checks"]["transcript_txt"] and not result["checks"]["audio_cached"]:
        result["status"] = "unprocessed"
        result["recommendation"] = "Run extract_transcript.py to process this video"

    return result


def _get_cache_size() -> float:
    """Calculate total cache size in MB."""
    if not CACHE_DIR.exists():
        return 0.0
    total = sum(f.stat().st_size for f in CACHE_DIR.rglob("*") if f.is_file())
    return round(total / (1024 * 1024), 1)


# ---------------------------------------------------------------------------
# CLI Dispatcher
# ---------------------------------------------------------------------------

COMMANDS = {
    "check-env": action_check_env,
    "list-transcripts": action_list_transcripts,
    "list": action_list_transcripts,
    "search": action_search,
    "stats": action_stats,
    "diagnose": action_diagnose,
}


def main():
    if len(sys.argv) < 2:
        print("Usage: python action.py <command> [args...]")
        print(f"Commands: {', '.join(COMMANDS.keys())}")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd not in COMMANDS:
        print(f"Unknown command: {cmd}")
        print(f"Available: {', '.join(COMMANDS.keys())}")
        sys.exit(1)

    fn = COMMANDS[cmd]
    args = sys.argv[2:]

    try:
        if cmd == "search":
            if not args:
                print("Usage: python action.py search <query>")
                sys.exit(1)
            result = fn(query=" ".join(args))
        elif cmd == "diagnose":
            if not args:
                print("Usage: python action.py diagnose <video_id>")
                sys.exit(1)
            result = fn(video_id=args[0])
        else:
            result = fn()

        print(json.dumps(result, ensure_ascii=False, indent=2))

    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
