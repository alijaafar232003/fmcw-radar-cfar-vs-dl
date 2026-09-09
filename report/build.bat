@echo off
REM Build the report. Regenerates numbers.tex first so no result can go stale.
setlocal
set PATH=%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64;%LOCALAPPDATA%\Programs\MiKTeX\executables\windows-x64\biber;%PATH%
pushd "%~dp0"
..\venv\Scripts\python.exe ..\scripts\generate_report_numbers.py
..\venv\Scripts\python.exe ..\scripts\make_appendix_tables.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
popd
endlocal
