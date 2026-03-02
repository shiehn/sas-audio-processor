<p align="center">
  <img src="assets/sas-audio-engine-graphic.png" alt="SAS Audio Processor" width="600">
</p>

# SAS Audio Processor

This is a component of [Signals & Sorcery](https://signalsandsorcery.com/), a music production application.

A suite of 25 audio processing tools — trim, normalize, compress, EQ, reverb, pitch-shift, time-stretch, key detection, MIDI extraction, and more — exposed as MCP tools via [DeclarAgent](https://github.com/shiehn/DeclarAgent).

## What You Get

| Category | Tools |
|----------|-------|
| **Processing** | trim, time-stretch, convert, mono, silence-remove, split-bars |
| **Effects** | normalize, gain, compress, eq, reverb, limit, filter, pitch-shift |
| **Analysis** | analyze, detect-key, loudness, onset-detect |
| **MIDI** | melody-to-midi |
| **Composite** | master-track, sample-prep, tempo-match, full-analysis, melody-extract |

Every tool accepts WAV files and outputs structured JSON.

## Quick Start (Claude Code)

This is the primary use case — use these audio tools directly from Claude Code via MCP.

### 1. Install DeclarAgent

```bash
go install github.com/shiehn/declaragent@latest
```

### 2. Install sas-audio-processor

```bash
# From source
git clone https://github.com/shiehn/sas-audio-processor.git
cd sas-audio-processor
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

Or use the pre-built binary (macOS only):

```bash
./build.sh
# Binary at dist/sas-processor/sas-processor
```

### 3. Add MCP config

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "sas-audio": {
      "command": "declaragent",
      "args": ["--plans-dir", "/path/to/sas-audio-processor/plans"]
    }
  }
}
```

### 4. Use it

Now Claude Code has access to all 25 audio tools. Example conversation:

```
You: Analyze this drum loop and tell me the BPM
Claude: [calls analyze tool] The drum loop is at 128 BPM, 44100 Hz, stereo, 8.2 seconds long.

You: Trim it to 4 bars and normalize to -14 LUFS
Claude: [calls trim tool, then normalize tool] Done — trimmed to 4 bars starting at the downbeat, normalized to -14 LUFS.

You: Now master it
Claude: [calls master-track tool] Applied normalize → compress → limit chain. Output saved.
```

## Quick Start (Other MCP Clients)

### Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "sas-audio": {
      "command": "declaragent",
      "args": ["--plans-dir", "/path/to/sas-audio-processor/plans"]
    }
  }
}
```

### Cursor / Windsurf / Copilot

Add the same MCP server config to your editor's MCP settings. The `command` and `args` are identical — only the config file location differs per editor.

## How It Works

```
MCP Client (Claude Code, Claude Desktop, etc.)
    │
    └─► DeclarAgent (reads YAML plan files from plans/)
            │
            └─► sas-processor <subcommand> --input ... --output ...
                    │
                    └─► stdout: JSON result
```

Each YAML plan file in `plans/` defines one MCP tool. DeclarAgent reads these at startup and exposes them to MCP clients. When a tool is called, DeclarAgent runs the `sas-processor` CLI under the hood.

## Available Tools

### Processing

| Tool | Description |
|------|-------------|
| `trim` | Detect downbeat and trim audio to specified number of bars |
| `time-stretch` | Change tempo while preserving pitch |
| `convert` | Change sample rate and/or bit depth |
| `mono` | Convert stereo to mono |
| `silence-remove` | Trim silence from start and end |
| `split-bars` | Split audio into N-bar chunks |

### Effects

| Tool | Description |
|------|-------------|
| `normalize` | Normalize to target LUFS or peak level |
| `gain` | Apply volume change in dB |
| `compress` | Dynamics compression with threshold/ratio/attack/release |
| `eq` | Parametric EQ band (boost or cut at frequency) |
| `reverb` | Algorithmic reverb with room size and wet/dry mix |
| `limit` | Brick-wall limiter |
| `filter` | High-pass or low-pass filter |
| `pitch-shift` | Shift pitch by semitones |

### Analysis

| Tool | Description |
|------|-------------|
| `analyze` | Detect BPM, duration, sample rate, channels |
| `detect-key` | Detect musical key and mode (major/minor) |
| `loudness` | Measure integrated LUFS and peak dB |
| `onset-detect` | Detect onset/transient times |

### MIDI

| Tool | Description |
|------|-------------|
| `melody-to-midi` | Extract monophonic melody to MIDI file |

### Composite (Multi-Step)

| Tool | Description |
|------|-------------|
| `master-track` | normalize → compress → limit |
| `sample-prep` | analyze → trim → normalize → convert |
| `tempo-match` | analyze → time-stretch to target BPM |
| `full-analysis` | analyze + detect-key + loudness |
| `melody-extract` | normalize → melody-to-midi |

## CLI Reference

All subcommands output line-delimited JSON to stdout. Errors go to stderr as JSON.

```bash
sas-processor <subcommand> [options]
```

### ping

```bash
sas-processor ping
```

Health check — returns version and status.

### analyze

```bash
sas-processor analyze --input <file>
```

Returns BPM, duration, sample rate, and channel count.

### trim

```bash
sas-processor trim --input <file> --output <file> --bpm <float> --bars <int> [--meter 4] [--verbose]
```

Detects the downbeat and trims audio to the specified number of bars.

### time-stretch

```bash
sas-processor time-stretch --input <file> --output <file> --source-bpm <float> --target-bpm <float>
```

Changes tempo while preserving pitch.

### normalize

```bash
sas-processor normalize --input <file> --output <file> [--mode lufs|peak] [--target-lufs -14] [--target-peak -1]
```

### gain

```bash
sas-processor gain --input <file> --output <file> --db <float>
```

### mono

```bash
sas-processor mono --input <file> --output <file>
```

### convert

```bash
sas-processor convert --input <file> --output <file> [--sample-rate <int>] [--bit-depth 16|24|32]
```

### silence-remove

```bash
sas-processor silence-remove --input <file> --output <file> [--top-db 30]
```

### compress

```bash
sas-processor compress --input <file> --output <file> [--threshold -20] [--ratio 4] [--attack 1.0] [--release 100]
```

### eq

```bash
sas-processor eq --input <file> --output <file> --freq <float> --gain-db <float> [--q 1.0]
```

### reverb

```bash
sas-processor reverb --input <file> --output <file> [--room-size 0.5] [--damping 0.5] [--wet-level 0.33]
```

### limit

```bash
sas-processor limit --input <file> --output <file> [--threshold -1]
```

### filter

```bash
sas-processor filter --input <file> --output <file> --type highpass|lowpass --cutoff-hz <float>
```

### pitch-shift

```bash
sas-processor pitch-shift --input <file> --output <file> --semitones <float>
```

### detect-key

```bash
sas-processor detect-key --input <file>
```

Returns key, mode (major/minor), and confidence score.

### loudness

```bash
sas-processor loudness --input <file>
```

Returns integrated LUFS and peak dB.

### onset-detect

```bash
sas-processor onset-detect --input <file>
```

Returns onset count and onset times in seconds.

### split-bars

```bash
sas-processor split-bars --input <file> --output-dir <dir> --bpm <float> [--bars-per-chunk 1] [--meter 4]
```

### melody-to-midi

```bash
sas-processor melody-to-midi --input <file> --output <file.mid>
```

## Building from Source

### Prerequisites

- macOS 10.15+
- Python 3.9+

### Build Steps

```bash
git clone https://github.com/shiehn/sas-audio-processor.git
cd sas-audio-processor
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt && pip install -e .

# Build self-contained binary
pyinstaller sas-processor.spec --clean --noconfirm
# Output: dist/sas-processor/
```

Or use the build script:

```bash
./build.sh
```

### Cross-Architecture Builds

**Apple Silicon (ARM64):**
```bash
pyinstaller sas-processor.spec --clean --noconfirm
mv dist/sas-processor dist/sas-processor-arm64
```

**Intel (x86_64):**
```bash
arch -x86_64 /bin/bash -c "
  python3 -m venv venv-x64
  source venv-x64/bin/activate
  pip install -r requirements.txt
  pyinstaller sas-processor.spec --clean --noconfirm
"
mv dist/sas-processor dist/sas-processor-x86_64
```

## Running Tests

```bash
source venv/bin/activate

# All tests
pytest tests/ -v

# E2E subcommand tests only
pytest tests/test_plans_e2e.py -v

# With coverage
pytest tests/ -v --cov=sas_processor
```

## JSON Output Format

All subcommands output line-delimited JSON to stdout.

### Result (stdout)

```json
{"type": "<subcommand>", "success": true, "output": "/path/to/output.wav", ...}
```

### Progress Events (stdout, trim only)

```json
{"type": "progress", "stage": "loading", "percent": 0}
{"type": "progress", "stage": "detecting", "percent": 30}
{"type": "progress", "stage": "trimming", "percent": 60}
{"type": "progress", "stage": "complete", "percent": 100}
```

### Errors (stderr)

```json
{"type": "error", "code": "FILE_NOT_FOUND", "message": "Input file not found: /path/to/file.wav", "severity": "fatal"}
```

## Limitations

- **WAV only** — no MP3, FLAC, or other formats
- **macOS only** — binary builds target macOS (CLI works anywhere Python runs)
- **Binary size ~220MB** — due to numpy/scipy/librosa dependencies

## License

MIT
