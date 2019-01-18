#!/usr/bin/env bash
echo "TEST pylint.sh"
echo $PATH
printf "\n"
echo $PYTHONPATH
pylint $@
return 1
