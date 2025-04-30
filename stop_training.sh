#!/bin/bash
pgrep -f train | xargs kill -9
pgrep -f tensorboard | xargs kill -9
