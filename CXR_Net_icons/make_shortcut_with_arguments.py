#!/usr/bin/env python
from pyshortcuts import make_shortcut
make_shortcut('/Users/dgatti/Documents/COVID19/CXR-Net_for_github/CXR-Net/run_CXR-Net/CXR_Net_gui.py --xdir /Users/dgatti/Documents/COVID19/CXR-Net_for_github/CXR-Net/run_CXR-Net',
              name='g_CXR_Net',
              description='CXR_Net gui',
              folder=None,
              icon='/Users/dgatti/Desktop/CXR_Net_icons/CXR_NET_ICON.icns',
              terminal=True, desktop=True, startmenu=True,
              executable='/Users/dgatti/venv_jupyter/bin/python3.8')