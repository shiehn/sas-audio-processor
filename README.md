# SAS Audio Processor

A self-contained macOS binary for beat detection and bar-aligned audio trimming. Designed to be called from an Electron application.

## Features

- **Beat Detection** - Uses librosa with a known BPM for accurate beat tracking
- **Downbeat Detection** - Energy-based algorithm finds the "1" of each bar
- **Bar-Aligned Trimming** - Cuts audio starting at a downbeat for exact bar counts
- **Self-Contained** - No external dependencies required (no Python, no FFmpeg)
- **JSON Output** - Progress events and results in JSON for easy parsing

## Quick Start

### Using the Binary

```bash
# Basic usage
./sas-processor --input song.wav --output trimmed.wav --bpm 120 --bars 8

# With JSON output (for Electron integration)
./sas-processor --input song.wav --output trimmed.wav --bpm 120 --bars 8 --json

# Verbose mode
./sas-processor --input song.wav --output trimmed.wav --bpm 120 --bars 8 --verbose
```

### CLI Options

| Option | Required | Description |
|--------|----------|-------------|
| `--input`, `-i` | Yes | Input WAV file path |
| `--output`, `-o` | Yes | Output WAV file path |
| `--bpm` | Yes | BPM of the audio (1-999) |
| `--bars` | Yes | Number of bars to extract (1-999) |
| `--meter` | No | Beats per bar (default: 4 for 4/4 time) |
| `--key` | No | Musical key (reserved for future use) |
| `--json` | No | Output progress and results as JSON |
| `--verbose`, `-v` | No | Verbose output |

## Building from Source

### Prerequisites

- macOS 10.15+
- Python 3.9+
- Virtual environment recommended

### Build Steps

```bash
# Clone the repository
git clone <repository-url>
cd sas-audio-processor

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Build the binary
pyinstaller sas-processor.spec --clean --noconfirm

# Binary output location
ls -la dist/sas-processor/
```

Or use the build script:

```bash
./build.sh
```

### Building for Different Architectures

**Apple Silicon (ARM64):**
```bash
# On an Apple Silicon Mac
pyinstaller sas-processor.spec --clean --noconfirm
mv dist/sas-processor dist/sas-processor-arm64
```

**Intel (x86_64):**
```bash
# On an Intel Mac, or using Rosetta on Apple Silicon:
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
# Activate virtual environment
source venv/bin/activate

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=sas_processor

# Run specific test file
pytest tests/test_processor.py -v
```

## JSON Output Format

When using `--json`, output is line-delimited JSON:

### Progress Events (stdout)
```json
{"type": "progress", "stage": "loading", "percent": 0}
{"type": "progress", "stage": "detecting", "percent": 30}
{"type": "progress", "stage": "trimming", "percent": 60}
{"type": "progress", "stage": "writing", "percent": 80}
{"type": "progress", "stage": "complete", "percent": 100}
```

### Final Result (stdout)
```json
{
  "type": "complete",
  "success": true,
  "output": "/path/to/output.wav",
  "downbeat_time": 0.523,
  "original_duration": 16.0,
  "output_duration": 8.0,
  "sample_rate": 44100
}
```

### Errors (stderr)
```json
{
  "type": "error",
  "code": "FILE_NOT_FOUND",
  "message": "Input file not found: /path/to/file.wav",
  "severity": "fatal"
}
```

## Electron Integration Example

```typescript
import { spawn } from 'child_process';
import * as path from 'path';

async function processAudio(
  inputPath: string,
  outputPath: string,
  bpm: number,
  bars: number
): Promise<{ success: boolean; downbeat_time: number }> {
  return new Promise((resolve, reject) => {
    const binaryPath = path.join(__dirname, 'binaries',
      process.arch === 'arm64' ? 'sas-processor-arm64' : 'sas-processor-x86_64',
      'sas-processor'
    );

    const proc = spawn(binaryPath, [
      '--input', inputPath,
      '--output', outputPath,
      '--bpm', bpm.toString(),
      '--bars', bars.toString(),
      '--json'
    ]);

    let result = '';

    proc.stdout.on('data', (data) => {
      const lines = data.toString().split('\n');
      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const json = JSON.parse(line);
          if (json.type === 'progress') {
            console.log(`Progress: ${json.stage} ${json.percent}%`);
          } else if (json.type === 'complete') {
            result = line;
          }
        } catch {}
      }
    });

    proc.on('close', (code) => {
      if (code === 0 && result) {
        resolve(JSON.parse(result));
      } else {
        reject(new Error(`Process exited with code ${code}`));
      }
    });
  });
}
```

## Limitations

- **WAV only** - Only WAV files are supported (no MP3, FLAC, etc.)
- **4/4 time** - Downbeat detection assumes 4/4 time signature
- **Binary size** - ~220MB due to scientific Python dependencies

## Technical Details

### Beat Detection
Uses `librosa.beat.beat_track()` with the provided BPM to skip tempo estimation and focus on beat placement.

### Downbeat Detection
Energy-based heuristic that:
1. Gets onset strength at each detected beat
2. Groups beats by bar (4 beats per bar in 4/4)
3. Finds which beat position has highest average energy across bars
4. Returns that position as the downbeat

### Audio Processing
- Preserves original sample rate and bit depth
- Uses numpy array slicing for efficient trimming
- Outputs via soundfile (libsndfile)

## License

MIT
