#!/bin/bash
nohup python manage.py runserver 10.144.10.191:2000 > log 2>&1 &
