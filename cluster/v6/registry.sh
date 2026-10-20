#!/bin/bash
cd /local/home/tianrgua/cvproject && SMOKE_REGISTRY_DIR=cluster/v6 exec .venv/bin/python cluster/v4/registry.py "$@"
