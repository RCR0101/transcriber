import os
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

# Main GUI Analysis
a = Analysis(
    ['gui.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['tkinter'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Transcriber Analysis (to bundle the transcriber with the GUI)
transcriber = Analysis(
    ['transcriber/cli.py'],
    pathex=[],
    binaries=[],
    datas=collect_data_files('whisper'),  # Include whisper model files
    hiddenimports=['whisper', 'transcriber.engine', 'transcriber.audio'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Merge analyses
MERGE((a, 'gui', 'gui'), (transcriber, 'transcriber', 'transcriber'))

# Create the GUI executable
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='TranscriberGUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to False for a windowed application
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
) 