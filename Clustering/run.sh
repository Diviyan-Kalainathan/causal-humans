#!/bin/bash
#set -euo pipefail

echo "Lancement du script batch"

for ((i = 0; i < 8; i++ )); do
	python main.py i 
done
