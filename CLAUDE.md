# CLAUDE.md - SAS Audio Processor

## Project Overview

This is a self-contained macOS binary for audio processing, designed to be called from an Electron app. It performs:
1. **Beat detection** using librosa with a known BPM
2. **Downbeat detection** using energy-based heuristics (finds beat 1 of the bar)
3. **Audio trimming** starting at the first downbeat for a specified number of bars

## Test Discipline (CRITICAL - NEVER SKIP)

**⚠️ ALWAYS run tests BEFORE and AFTER every code change. NEVER ignore test failures. ⚠️**

### The Rules
1. **BEFORE starting work**: Run the full test suite to establish a baseline. If tests are already failing, STOP and notify the user before proceeding.
2. **AFTER every change**: Run the full test suite again. All tests must pass.
3. **NEVER ignore failures**: A failing test is a BLOCKER. Do not move on, do not commit, do not consider the work done until all tests pass.
4. **No silent regressions**: If your change breaks an existing test, fix it immediately. Do not assume the test is wrong — investigate first.
5. **Add tests for new code**: Any significant new functionality must include test coverage.

### Test Commands
```bash
cd /Users/stevehiehn/sas-platform/sas-audio-processor && source venv/bin/activate && pytest tests/ -v
```

---

## Rebuild & Stage Rule (CRITICAL - ALWAYS REBUILD AFTER CHANGES)

**⚠️ After ANY code change here, you MUST rebuild the PyInstaller binary and stage it into `sas-assistant/resources/`. Skipping this means `npm run` / `npm run dev` / `npm run release` in sas-assistant will silently use the OLD binary, and your change will appear to have no effect. ⚠️**

This is the #1 cause of "I fixed the bug but it still happens" in this repo. Tests passing here is necessary but NOT sufficient — the Electron app spawns the pre-built `sas-processor` binary from `resources/`, not the Python source.

### Required after every change

```bash
cd ../sas-assistant && npm run build:native:processor
```

This builds for the current architecture and copies into `sas-assistant/resources/audio-processor/{arm64,x86_64}/`.

To verify the staged binary is present and valid:
```bash
cd ../sas-assistant && npm run verify-resources
```

For a release (both architectures):
```bash
cd ../sas-assistant && npm run build:native   # all natives, current arch
# or `npm run release` which handles both arm64 and x86_64
```

### Workflow checklist
1. Make change in `sas-audio-processor`
2. Run tests here (per above)
3. **Rebuild + stage** via `cd ../sas-assistant && npm run build:native:processor`
4. Test in the Electron app (`cd ../sas-assistant && npm run dev`)
5. Do NOT consider the change done until step 3 has run.

---

## Architecture

```
Electron App
    │
    └─► spawn('sas-processor --input ... --bpm 120 --bars 8 --output ...')
        │
        └─► stdout: JSON progress events + final result
```

## Key Technical Decisions

- **librosa** for beat detection (with known BPM to skip tempo estimation)
- **soundfile** for WAV I/O (bundled with librosa, no FFmpeg needed)
- **PyInstaller** for creating self-contained binary
- **numpy < 2.0** required (PyInstaller compatibility)
- **optimize=0** in PyInstaller spec (numpy docstring compatibility)

## Project Structure

```
sas-audio-processor/
├── src/sas_processor/
│   ├── __init__.py
│   ├── __main__.py         # Entry point for python -m sas_processor
│   ├── cli.py              # Click-based CLI
│   ├── processor.py        # Core processing logic
│   └── beat_detection.py   # librosa beat/downbeat detection
├── tests/
│   └── test_processor.py   # pytest tests
├── pyproject.toml          # Project config
├── requirements.txt        # Dependencies
├── sas-processor.spec      # PyInstaller config
└── build.sh               # Build script
```

## Building

### Prerequisites
- Python 3.9+ (macOS system Python works)
- Virtual environment

### Quick Build (current architecture)
```bash
./build.sh
```

### Manual Build
```bash
# Create venv
python3 -m venv venv
source venv/bin/activate

# Install deps
pip install -r requirements.txt
pip install -e .

# Build
pyinstaller sas-processor.spec --clean --noconfirm

# Output: dist/sas-processor/
```

### Building for Different Architectures

**ARM64 (Apple Silicon):**
```bash
# On an Apple Silicon Mac:
pyinstaller sas-processor.spec --clean --noconfirm
mv dist/sas-processor dist/sas-processor-arm64
```

**x86_64 (Intel):**
```bash
# On Intel Mac OR using Rosetta:
arch -x86_64 /bin/bash -c "python3 -m venv venv-x64 && source venv-x64/bin/activate && pip install -r requirements.txt && pyinstaller sas-processor.spec --clean --noconfirm"
mv dist/sas-processor dist/sas-processor-x86_64
```

## Testing

```bash
# Run all tests
source venv/bin/activate
pytest tests/ -v

# Test CLI directly
sas-processor --input test.wav --output out.wav --bpm 120 --bars 4 --json
```

## CLI Interface

```bash
sas-processor \
  --input /path/to/audio.wav \
  --output /path/to/output.wav \
  --bpm 120 \
  --bars 8 \
  [--meter 4] \
  [--key Am] \
  [--json] \
  [--verbose]
```

### JSON Output Format
```json
{"type": "progress", "stage": "loading", "percent": 0}
{"type": "progress", "stage": "detecting", "percent": 30}
{"type": "progress", "stage": "trimming", "percent": 60}
{"type": "progress", "stage": "writing", "percent": 80}
{"type": "progress", "stage": "complete", "percent": 100}
{"type": "complete", "success": true, "output": "/path/output.wav", "downbeat_time": 0.523, ...}
```

### Error Format (stderr)
```json
{"type": "error", "code": "FILE_NOT_FOUND", "message": "Input file not found", "severity": "fatal"}
```

## Key Files

- `src/sas_processor/beat_detection.py` - Core beat/downbeat detection algorithms
- `src/sas_processor/processor.py` - Main processing pipeline
- `src/sas_processor/cli.py` - Command-line interface
- `sas-processor.spec` - PyInstaller build configuration

## Important Notes

1. **WAV only** - Only WAV files are supported (no MP3/FLAC)
2. **Preserves format** - Output maintains original sample rate and bit depth
3. **Binary size ~220MB** - Due to numpy/scipy/librosa dependencies
4. **No external deps** - Binary is fully self-contained
5. **4/4 time assumed** - Downbeat detection assumes 4/4 time signature

## Downbeat Detection Algorithm

1. Detect all beats using librosa with known BPM
2. Get onset strength at each beat position
3. Group beats by 4 (for 4/4 time)
4. Find which beat position (1, 2, 3, or 4) has highest average energy
5. That's the downbeat position

## Troubleshooting

### Binary crashes on startup
- Ensure numpy < 2.0 in requirements.txt
- Ensure optimize=0 in spec file

### Beat detection inaccurate
- Verify correct BPM is provided
- Consider upgrading to madmom for better downbeat detection

### Large binary size
- This is expected (~220MB)
- Can reduce by excluding unused scipy/sklearn modules
