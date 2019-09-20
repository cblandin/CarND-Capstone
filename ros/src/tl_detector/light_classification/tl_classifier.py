from styx_msgs.msg import TrafficLight
import numpy as np
import ropsy
import cv2
from imutils import *




class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        #pass
        
        self.model = tf.Graph()
        with self.model.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile("/home/student/Documents/CarND-Capston/ros/src/tl_detector/light_classification/graph_optimized.pb",'rb') as fileID:
                serialized_graph = fid.read()
                graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(graph_def, name='')
        self.sess = tf.Session()

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        height, width = image.shape[:2]
        
        if height > 1000 or width > 1000:
            if height > width:
                image = imutils.resize(image, height = 1000)
            else:
                image = imutils.resize(image, width = 1000)  

        
        with self.model.as_default():
            image_tensor = graph.get_tensor_by_name('image_tensor:0')
            boxes_tensor = graph.get_tensor_by_name('detection_boxes:0')
            scores_tensor = graph.get_tensor_by_name('detection_scores:0')
            classes_tensor = graph.get_tensor_by_name('detection_classes:0')
            detections_tensor = graph.get_tensor_by_name('num_detections:0')
            
            
            N, boxes, scores, labels = self.sess.run([detections_tensor, boxes_tensor, scores_tensor, classes_tensor], feed_dict = {image_tensor:image})-=
        
        
        scores = np.squeeze(scores)
        labels = np.squeeze(labels)
        
        
        max_i = np.argmax(scores)
        
        
        if scores[max_i] > 0.7:
            if labels[max_i] == 0:
                return TrafficLight.GREENLIGHT
            elif labels[max_i] == 1:
                return TrafficLight.YELLOWLIGHT
            elif labels[max_i] = 2:
                return TrafficLight.REDLIGHT
        
        
        return TrafficLight.UNKNOWN
