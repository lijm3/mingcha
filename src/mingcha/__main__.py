"""python -m mingcha 入口（等价于 console_scripts 的 mingcha / mc）。"""
from .cli import main

raise SystemExit(main())
