@echo off

if "%~1"=="" exit /b 1

set "file=%~1"

pyinstaller --onefile --distpath ./bin "%file%"

for %%f in ("%file%") do set "name=%%~nf"

rmdir /s /q build
del /q "%name%.spec"