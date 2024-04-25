All of this code was tested using Ubuntu on WSL

Set up a virtualenv
Make sure graphics drivers are up to date, including CUDNN
Install requirements.txt
Make sure latest version of torch from this website is downloaded for the correct computer: https://pytorch.org/get-started/locally/



YOLOv8:

Run all cells in YOLOv8.ipynb
During the process, the Pascal-VOC-2012-1 folder will include the YOLO version of the dataset

output will be in the 'runs' folder as train{number} where number is the highest next number

R-CNN:

Run all cells in ResNet.ipynb. This will output to runs folder 

Use the following command in terminal and launch the localhost output on a browser to see the results
tensorboard --logdir=./


SSD

Open ssd.ipynb
Make sure that the version of Pascal VOC in the folder is the COCO version, may need to rename the files in cell 3

run all cells in ssd.ipynb


RCNN- fewshot

Make sure you have the tensor dataset for Pascal-VOC-2012 
Run all cells in ResNetFSL.ipynb
