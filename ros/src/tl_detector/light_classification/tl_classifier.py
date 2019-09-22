from styx_msgs.msg import TrafficLight
import numpy as np
import rospy
import cv2
import tensorflow as tf



class TLClassifier(object):
    def __init__(self):
        #load classifier
        
        self.model = tf.Graph()
        with self.model.as_default():
            graph_def = tf.GraphDef()
            #Using model that has been trained by  Marco Marasca that can be found here: https://github.com/marcomarasca/SDCND-Traffic-Light-Detection/tree/master/saved_models/converted_optimized
            with tf.gfile.GFile("/home/student/Documents/CarND-Capstone/ros/src/tl_detector/light_classification/graph_optimized.pb",'rb') as fileID:
                serialized_graph = fileID.read()
                graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(graph_def, name='')
            self.sess = tf.Session() # create session once, not every time we get an image

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #implement light color prediction
        
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        height, width = image.shape[:2]
        
        # scale down while maintaining aspect ratio if necessary
        if height > 1000 or width > 1000:
            if height > width:
                image = cv2.resize(image, (1000,int(width*1000/height)))
            else:
                image = cv2.resize(image,(int(height*1000/width), 1000))

        image = [image] #Add dimension
        
        #load inputs
        image_tensor = self.model.get_tensor_by_name('image_tensor:0')
        boxes_tensor = self.model.get_tensor_by_name('detection_boxes:0')
        scores_tensor = self.model.get_tensor_by_name('detection_scores:0')
        classes_tensor = self.model.get_tensor_by_name('detection_classes:0')
        detections_tensor = self.model.get_tensor_by_name('num_detections:0')
            
        #run through model
        with self.model.as_default():
            N, boxes, scores, labels = self.sess.run([detections_tensor, boxes_tensor, scores_tensor, classes_tensor], feed_dict = {image_tensor:image})
        
        
        #make 1-D
        scores = np.squeeze(scores)
        labels = np.squeeze(labels)
        
        
        #Find Max score
        max_i = np.argmax(scores)
        
#	rospy.logwarn("Traffic Light: {0}".format(labels[max_i]))        
        #convert from model to TrafficLight messages if score is higher than .7
        if scores[max_i] > 0.7:
            if labels[max_i] == 1:
                return TrafficLight.GREEN
            elif labels[max_i] == 2:
                return TrafficLight.YELLOW
            elif labels[max_i] == 3:
                return TrafficLight.RED
        
        
        return TrafficLight.UNKNOWN
