#!/bin/bash
pgrep -f train | xargs kill -9
pgrep -f tensorboard | xargs kill -9
sleep 10
pgrep -f tensorboard | xargs kill -9
ps -ef | grep tensor
ps -ef | grep train
