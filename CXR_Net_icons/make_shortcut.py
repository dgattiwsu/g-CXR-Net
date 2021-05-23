#!/usr/bin/env python
from pyshortcuts import make_shortcut
make_shortcut('/Users/dgatti/Desktop/g-CXR-Net/g_CXR_Net.py --xdir /Users/dgatti/Desktop/g-CXR-Net',
              name='g_CXR_Net',
              description='CXR_Net GUI app',
              folder='/Users/dgatti/Desktop/g-CXR-Net',
              icon='/Users/dgatti/Desktop/g-CXR-Net/CXR_Net_icons/CXR_NET_ICON.icns',
              terminal=True, desktop=True, startmenu=True,
              executable='/Users/dgatti/venv_jupyter/bin/python3.8')