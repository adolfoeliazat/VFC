# VFC
Video-Frames-Classifier

VFC is an example of using a pretrained neural network to check frames from video for presence of specific classes of objects. It was designed to be used on multiple camera-enabled systems operating on a common network. A neural network is loaded into memory (in this case the resnet_50_weights for theano as in this Keras example: https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py), then threads are spawned to listen for UDP multicast messages on one or more ports as defined in "hostnames_ports_listen.csv" (this system's hostname must preceed one or more valid port numbers). Valid messages may contain two fields; the first is the command. If the command is "predict" then the second field is taken to be the file path of the video to analyze. The video frames are then input sequentially to the neural network which outputs probabilities for each of the 1000 classes in ImageNet. These class probabilities are used determine the likelihood that a vehicle (as defined in "vehicle_words.csv") is present in each image. A vehicle (car, truck, motorcycle, etc.) is considered present if a class in "vehicle_words.csv" is present in the top 20 predicted classes and has a probability > 0.09.  

The code is run from the command-line like:  
`$ python video_frames_classifier.py '224.1.1.1'`  

The code to send the UDP messages is "send_UDP_messages_to_listeners_multicast.py". This should be run from a system on the same network (or even another terminal on the same system) as follows:  
`$ python send_UDP_message_to_listeners_multicast.py predict outputTest.avi`  
 where "predict" is the commnd and "outputTest.avi" is the videofile to be analyzed.  

Requirements:  
python 3.x  
theano  
keras  
skimage  
imageio  

also requires that the model weights "resnet50_weights_th_dim_ordering_th_kernels.h5" be in the Keras directory ~/.keras/models AND that the two CSV files "vehicle_words.csv" and "hostnames_ports_listen.csv" be in the VFC directory.  
