# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for SAS Audio Processor.

Build commands:
  # Build for current architecture (recommended)
  pyinstaller sas-processor.spec

  # Build specifically for arm64 (Apple Silicon)
  pyinstaller sas-processor.spec --target-arch arm64

  # Build specifically for x86_64 (Intel)
  pyinstaller sas-processor.spec --target-arch x86_64
"""

import sys
from pathlib import Path

# Get the spec file directory
spec_dir = Path(SPECPATH)

a = Analysis(
    [str(spec_dir / 'src' / 'sas_processor' / 'cli.py')],
    pathex=[str(spec_dir / 'src')],
    binaries=[],
    datas=[],
    hiddenimports=[
        # Librosa and dependencies
        'librosa',
        'librosa.core',
        'librosa.beat',
        'librosa.onset',
        'librosa.feature',
        'librosa.util',
        # Numba JIT compiler (required by librosa)
        'numba',
        'numba.core',
        'numba.cpython',
        'llvmlite',
        'llvmlite.binding',
        # Audio I/O
        'soundfile',
        'audioread',
        # Scientific computing
        'numpy',
        'scipy',
        'scipy.signal',
        'scipy.fft',
        'scipy.ndimage',
        # Other dependencies
        'sklearn',
        'sklearn.utils',
        'joblib',
        'pooch',
        'resampy',
        'decorator',
        'packaging',
        'platformdirs',
        # Audio effects
        'pedalboard',
        'pyloudnorm',
        # MIDI extraction
        'basic_pitch',
        'basic_pitch.inference',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary modules to reduce size
        'tkinter',
        'matplotlib',
        'PIL',
        'IPython',
        'jupyter',
        'notebook',
        'sphinx',
        'docutils',
        'pygments',
        'pytest',
        'setuptools',
        'wheel',
        'pip',
    ],
    noarchive=False,
    optimize=0,  # Don't optimize - numpy docstrings require this
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='sas-processor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,  # Will use current arch or --target-arch flag
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=True,
    upx=True,
    upx_exclude=[],
    name='sas-processor',
)
