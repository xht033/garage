#!/usr/bin/env bash
echo "TEST pylint.sh"
echo $PATH
printf "\n"
echo $PYTHONPATH
echo "PIP freeze"
echo $@
pylint $@
echo "TEST import"
python scripts/test_import.py
return 1
