#!/bin/bash
gunicorn --workers 1 --bind 0.0.0.0:$PORT --timeout 300 app:app
