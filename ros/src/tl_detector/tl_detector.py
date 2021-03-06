#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
from scipy.spatial import KDTree
import tf
import cv2
import yaml

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.has_image = False
        self.lights = []


        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.waypoints = None
        self.waypoint_tree = None
        #rospy.logwarn("init waypoint_tree: {0}".format(self.waypoint_tree))
        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)
        self.loop()
       # rospy.spin()

    def loop(self):
	rate = rospy.Rate(4)
	while not rospy.is_shutdown():
#		rospy.logwarn("LOOP TL_DETECTOR {0}".format(self.waypoints is None))
		if self.has_image and self.waypoint_tree is not None and self.pose is not None and self.waypoints is not None:
#			rospy.logwarn("Process_Lights")
			light_wp, state = self.process_traffic_lights()
			self.upcoming_red_light_pub.publish(Int32(light_wp))
#			rospy.logwarn("Light_WP {0}".format(light_wp))
		rate.sleep()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        #rospy.logwarn("waypoint tree:{0}".format(self.waypoint_tree))
        if self.waypoint_tree is None:
            #self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree([[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints])

    def traffic_cb(self, msg):
#	rospy.logwarn("Traffic Lights Recieved")
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        
        # For latency issues when camera is turned on, only classify every 5 images
        if self.state_count == 0:
            self.has_image = True
            self.camera_image = msg
#            light_wp, state = self.process_traffic_lights()
        else:
            self.has_image = False
        self.state_count = self.state_count + 1
        if self.state_count > 5:
            self.state_count = 0



        
    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        # implement
        return self.waypoint_tree.query([x,y],1)[1]

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        return self.light_classifier.get_classification(cv_image)
        
        #return light.state
        
    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
	
        closest_light = None
        line_wp_ndx = None
        
        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose and self.waypoint_tree):
            car_wp_ndx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)

            # find the closest visible traffic light (if one exists)
            diff = len(self.waypoints.waypoints)
            for i, light in enumerate(self.lights):
                line = stop_line_positions[i]  #Get stop line waypoint index
                temp_wp_ndx = self.get_closest_waypoint(line[0],line[1])
                delta_ndx = temp_wp_ndx - car_wp_ndx
                if delta_ndx >= 0 and delta_ndx < diff: #if it is closer and in front of car, update
                    diff = delta_ndx
                    closest_light = light
                    line_wp_ndx = temp_wp_ndx
	#rospy.logwarn("Traffic Light Returned")
        if closest_light is not None:
            state = self.get_light_state(closest_light)
	if state == 0 or state == 1:
#		rospy.logwarn("Traffic state: {0}".format(line_wp_ndx)) 
            return line_wp_ndx, state
        else:
            return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
