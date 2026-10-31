@echo off
rem Dupla kattintassal indithato: run.ps1 futtatasa (ExecutionPolicy Bypass).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
if errorlevel 1 pause
