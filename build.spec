# build.spec
# Run with: pyinstaller build.spec
block_cipher = None

a = Analysis(
    ['transcriber/cli.py'],
    pathex=['.'],
    hiddenimports=[],
    datas=[            # ship Whisper models & ffmpeg
        ('models', 'whisper/models'),
    ],
    binaries=[
        ('/usr/local/bin/ffmpeg', '.'),  # adjust for Windows path
    ],
)

pyz  = PYZ(a.pure,  a.zipped_data, cipher=block_cipher)
exe  = EXE(pyz,
           a.scripts,
           name='transcriber',
           console=True,
           onefile=True)