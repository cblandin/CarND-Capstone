#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
import numpy as np

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 50 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.stopline_wp_ndx = -1
        self.max_decel = rospy.get_param('~decel_limit', -5)
        self.loop() 
        
    def loop(self):
        rate = rospy.Rate(4)
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints and self.waypoint_tree: # then get closest waypoints
                closest_waypoint_ndx = self.get_closest_waypoint_ndx()
                self.publish_waypoints(closest_waypoint_ndx)
            rate.sleep()
            
    def get_closest_waypoint_ndx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_ndx = self.waypoint_tree.query([x,y],1)[1]
        
        closest_coord = np.array(self.waypoints_2d[closest_ndx])
        prev_coord = np.array(self.waypoints_2d[closest_ndx - 1])
        current_pos = np.array([x,y])
        
        dot_prod = np.dot(closest_coord - prev_coord, current_pos - closest_coord)
        if dot_prod > 0:
            closest_ndx = (closest_ndx + 1) % len(self.waypoints_2d)
            
        return closest_ndx
        
    def publish_waypoints(self, closest_ndx):
        lane = self.generate_lane(closest_ndx)
        #lane.header = self.base_waypoints.header
        self.final_waypoints_pub.publish(lane)
        
    def generate_lane(self,closest_ndx):
        lane = Lane()
        farthest_ndx = closest_ndx + LOOKAHEAD_WPS
        base_waypoints = self.base_waypoints.waypoints[closest_ndx:closest_ndx + LOOKAHEAD_WPS]
#	rospy.logwarn("Stopline_wp_ndx : {0}".format(self.stopline_wp_ndx))
        if self.stopline_wp_ndx == -1 or (self.stopline_wp_ndx >= farthest_ndx):
            lane.waypoints = base_waypoints
        else:
#	    rospy.logwarn("DECEL")
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_ndx)
        
        return lane
        
        
    def decelerate_waypoints(self, waypoints, closest_ndx):
        
        stop_ndx = max(self.stopline_wp_ndx - closest_ndx -8, 0)
        old_vel = waypoints[stop_ndx].twist.twist.linear.x
        
        dist_for_decel = abs(old_vel*old_vel/(2.0*self.max_decel * 0.6)) # distance for linear decel to zero at 60% of MAX_DECEL
        new_waypoints = []
        
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose
            
            dist = self.distance(waypoints, i, stop_ndx) #distance to stoplight
            vel = (dist-dist_for_decel)/dist_for_decel*old_vel + old_vel
#	    rospy.logwarn("Vel : {0}".format(dist_for_decel))
            if vel < 0.25:
                vel = 0.0
            p.twist.twist.linear.x = min(vel, old_vel)
            new_waypoints.append(p)
            
        return new_waypoints
        
    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints
        if not self.waypoint_tree:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)
	    #rospy.logwarn("KD Tree Set")

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        #rospy.logwarn("Stop_light_received: {0}".format(msg.data))
        self.stopline_wp_ndx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
