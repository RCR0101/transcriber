import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

block_cipher = None

# Main GUI Analysis
a = Analysis(
    ['gui.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('transcriber', 'transcriber'),  # Include the entire transcriber module
        *collect_data_files('whisper'),  # Include whisper data files
        *collect_data_files('tkinter'),  # Include tkinter data files
    ],
    hiddenimports=[
        'tkinter',
        'tkinter.ttk',
        'tkinter.messagebox',
        'tkinter.filedialog',
        '_tkinter',
        'transcriber',
        'transcriber.engine',
        'transcriber.audio',
        'whisper',
        'numpy',
        'numpy.core',
        'numpy.lib',
        'numpy.linspace',
        'torch',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

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