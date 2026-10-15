# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Student AI Assistant console EXE.

Build with:  pyinstaller StudentAIAssistant.spec --noconfirm
"""
import os
from pathlib import Path

block_cipher = None

ROOT = Path(SPECPATH)

# Bundled immutable runtime assets (NO .env, memory, chat logs, git data)
datas = [
    (str(ROOT / "data" / "intents.json"), "data"),
    (str(ROOT / "data" / "offline_kb.json"), "data"),
    (str(ROOT / "models" / "vectorizer.pkl"), "models"),
    (str(ROOT / "models" / "intent_model.pkl"), "models"),
    (str(ROOT / "webapp" / "templates"), "webapp/templates"),
    (str(ROOT / "webapp" / "static"), "webapp/static"),
]

# Modules referenced dynamically or by dotted import path
hiddenimports = [
    "utils.paths",
    "utils.filelock",
    "services.response_policy",
    "nlp.preprocessing",
    "nlp.classifier",
    "features.commands",
    "utils.logger",
]

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "pytest",
        "jupyter",
        "matplotlib",
        "pandas",
        # python-dotenv's optional IPython integration drags in a huge
        # interactive stack that the packaged app never uses
        "dotenv.ipython",
        "IPython",
        "jedi",
        "black",
        "blib2to3",
        "prompt_toolkit",
        "lxml",
        "nbformat",
        "nbconvert",
        "PIL",
        "pygments",
        "astroid",
        "segno",
        "regex",
        "setuptools",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="StudentAIAssistant",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="StudentAIAssistant",
)