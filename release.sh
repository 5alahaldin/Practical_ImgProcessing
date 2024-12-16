#!/bin/bash

[ -z "$1" ] && { exit 1; }

file=$1

pyinstaller --onefile --distpath ./bin "$file"

name=$(basename "$file")
rm -rf build/ "${name%.*}.spec"