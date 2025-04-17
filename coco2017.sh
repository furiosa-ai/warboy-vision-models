# https://github.com/linzhenyuyuchen/Dataset-Download/blob/master/coco/coco2017.sh
#!/bin/bash

start=`date +%s`

DIR=datasets/coco
if [ ! -d "$DIR" ]; then
    mkdir -p $DIR
fi

cd "$DIR"

# Download the image data.
echo "Downloading MSCOCO val images ..."
wget http://images.cocodataset.org/zips/val2017.zip

# Download the annotation data.
echo "Downloading MSCOCO train/val annotations ..."
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

echo "Finished downloading. Now extracting ..."

# Unzip data
echo "Extracting val images ..."
unzip -q val2017.zip
echo "Extracting annotations ..."
unzip -q ./annotations_trainval2017.zip

echo "Removing zip files ..."
rm val2017.zip
rm annotations_trainval2017.zip

end=`date +%s`
runtime=$((end-start))

cd -

echo "Completed in " $runtime " seconds"
