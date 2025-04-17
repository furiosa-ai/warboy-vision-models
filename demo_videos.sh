#!/bin/bash

pip install gdown
sudo apt install unzip

DIR=datasets/demo_videos
if [ ! -d "$DIR" ]; then
    mkdir -p $DIR
fi

cd "$DIR"

echo "Downloading warboy tutorial files..."
gdown --id 17Ok3bAit8uFH1QO-TtZAlZpKvzS4ANCk

echo "Extracting warboy tutorial files..."
unzip warboy_tutorial.zip

echo "Moving demo videos from warboy tutorial files..."
mv warboy_tutorial/part5/assets/videos/detection_videos .
mv warboy_tutorial/part5/assets/videos/pose_videos .

echo "Removing warboy tutorial files..."
rm warboy_tutorial.zip
rm -rf warboy_tutorial
rm -rf __MACOSX

cd -

rm -rf __MACOSX
