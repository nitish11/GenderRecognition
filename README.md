# GenderRecognition
Caffe Model to detect Gender

## models folder contains
Models can be downloaded from [here](http://www.openu.ac.il/home/hassner/projects/cnn_agegender/cnn_age_gender_models_and_data.0.0.2.zip)
- age_gender_mean.t7
- haarcascade_frontalface_default.xml
- deploy_gender.prototxt  
- gender_net.caffemodel       
- mean.binaryproto

## How to use
- Caffe Installation [here](http://caffe.berkeleyvision.org/installation.html)
- Torch Installation with Lua[here](http://torch.ch/docs/getting-started.html#installing-torch)
- Other installations (Run 'luarocks install dpnn', 'pip install lutorpy')
- Download the trained model zip file from the link above and unzip the file
- Put all the model files in folder 'models'
- keep the codes and 'models' in the same location
- Make sure Camera is connected as USBID is 0(default)
- To check different implementation, run the following codes
* python gender_demo_lua.py : Python wrapper to run Lua code, fastest code
* th gender_demo_torch.lua  : Lua code to detect gender, best accuracy 
* python gender_demo_caffe.py : Caffe code, slower



## Lua implementation source : 
https://github.com/szagoruyko/torch-opencv-demos/blob/master/age_gender/demo.lua

## Caffe implementation  source:  
http://nbviewer.jupyter.org/url/www.openu.ac.il/home/hassner/projects/cnn_agegender/cnn_age_gender_demo.ipynb

## References
- https://gist.github.com/GilLevi/c9e99062283c719c03de
- http://www.openu.ac.il/home/hassner/projects/cnn_agegender/
- http://www.openu.ac.il/home/hassner/projects/cnn_agegender/cnn_age_gender_models_and_data.0.0.2.zip

