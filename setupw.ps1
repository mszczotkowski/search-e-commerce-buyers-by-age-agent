$AppHome=$PSScriptRoot
uv venv "$AppHome\.venv" --clear
& "$AppHome\.venv\Scripts\Activate.ps1"
uv sync
uv lock