# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for OpenGTO Backend.
Bundles the Flask API server with the neural network model.
"""

import os
import sys

block_cipher = None

# Get the project root
project_root = os.path.dirname(os.path.abspath(SPEC))

# Data files to include
datas = [
    # Include the trained model (only the final one to keep size manageable)
    (os.path.join(project_root, 'checkpoints_improved', 'gto_trainer_final.pt'),
     'checkpoints_improved'),
    # Include the src package
    (os.path.join(project_root, 'src'), 'src'),
]

# Hidden imports that PyInstaller might miss
hiddenimports = [
    'flask',
    'flask_cors',
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'numpy',
    # All src modules
    'src',
    'src.card',
    'src.cfr_engine',
    'src.curriculum',
    'src.deep_cfr',
    'src.evaluation',
    'src.game_state',
    'src.hand_utils',
    'src.information_set',
    'src.memory_buffer',
    'src.neural_network',
    'src.poker_engine',
    'src.self_play',
    'src.showdown',
    'src.trainer',
    'src.trainer_interface',
]

a = Analysis(
    [os.path.join(project_root, 'api_server.py')],
    pathex=[project_root],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'PIL',
        'scipy',
        'pandas',
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
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='opengto_backend',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Keep console for debugging, can set to False later
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(project_root, 'frontend', 'build', 'icon.ico') if os.path.exists(os.path.join(project_root, 'frontend', 'build', 'icon.ico')) else None,
)
