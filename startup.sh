#!/bin/bash
gunicorn -b 0.0.0.0:${PORT-8000} --workers 2 --threads 4 app:app
