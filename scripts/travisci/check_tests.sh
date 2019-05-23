#!/usr/bin/env bash
set -e

pip install nose2-timer
coverage run -m nose2 -c setup.cfg -v --with-id -E 'not cron_job and not huge and not flaky' --timer
coverage xml
bash <(curl -s https://codecov.io/bash)
