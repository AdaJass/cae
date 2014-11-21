#!/bin/sh

now="$(date): update all"

git add -A README.md
git add -A update.sh
git add -A cae_test.py
git add -A mlp_test.py
git add -A mlp_model.py
git add -A cae_tools.py
git add -A dlt_utils.py
git add -A conv_aes/*.py

git commit -m "$now"

git push origin master
