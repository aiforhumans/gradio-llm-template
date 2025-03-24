@echo off
python -m venv .venv
call .venv\Scripts\activate
pip install -r requirements.txt
echo Virtual environment set up and dependencies installed.
pause
