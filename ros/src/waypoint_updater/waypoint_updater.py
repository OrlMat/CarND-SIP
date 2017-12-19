#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from copy import deepcopy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32

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
MAX_DEACCELERATION = -0.2

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')
        print "Initial startup"

        rospy.Subscriber('current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('traffic_waypoint', Int32, self.traffic_cb)
        print "Have subscribed to current_pose, base_waypoints and traffic_waypoint"

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.baseWaypoints = None
        self.currentPose = None
        self.trafficLight = -1

        self.maxSpeed = rospy.get_param('/waypoint_loader/velocity')
        self.maxSpeed *= 1000.0/3600.0 #Changing from kph to m/s
        print "MaxSpeed is: " + str(self.maxSpeed)

        self.loop()

    def loop(self):

        print "Entering loop function"
        rate = rospy.Rate(5) # 5Hz
        while not rospy.is_shutdown():
            passedTrafficLight = False
            if (self.baseWaypoints is not None) and (self.currentPose is not None):
                #print "baseWaypoint and currentPose received"
                nextMessage = Lane()
                nextWaypointIndex = self.nextWaypoint(self.baseWaypoints)
                #print "nextWaypointIndex is: " + str(nextWaypointIndex)
                #print "TrafficLight is: " + str(self.trafficLight)

                waypointSet = self.baseWaypoints[nextWaypointIndex:nextWaypointIndex+LOOKAHEAD_WPS]
                for i, wp in enumerate(waypointSet):
                    newWp = deepcopy(wp)
                    index = nextWaypointIndex + i

                    newWp.twist.twist.linear.x = min(self.maxSpeed, newWp.twist.twist.linear.x)

                    if index >= self.trafficLight - 15 and not self.trafficLight == -1:
                        passedTrafficLight = True

                    if passedTrafficLight:
                        newWp.twist.twist.linear.x = 0

                    nextMessage.waypoints.append(newWp)

                '''
                Fixes all velocities differences between neighbouring waypoints
                 to make sure that we keep within acceleration limits before stop signs.

                This is done by starting at the last waypoint and looping over
                the set in reverse while making sure that the
                acceleration (braking) needed to reach the
                correct velocities is -3m/s² or less (more? due to negative sign).
                '''

                if passedTrafficLight:
                    lastWp = nextMessage.waypoints[-1]
                    for i, wp in reversed(list(enumerate(nextMessage.waypoints))):
                        dist = self.distance2(wp, lastWp)
                        lastSpeed = self.get_waypoint_velocity(lastWp)
                        currentSpeed = self.get_waypoint_velocity(wp)
                        if lastSpeed < 0.1:
                            speedBasedOnLastSpeed = lastSpeed + math.sqrt(-2 * MAX_DEACCELERATION * dist)
                        else:
                            temp = MAX_DEACCELERATION / (2 * dist)
                            speedBasedOnLastSpeed = lastSpeed * (1 - temp) / (1 + temp)
                        print "dist: " + str(dist) + " - lastSpeed: " + str(lastSpeed) + \
                        " - currentSpeed: " + str(currentSpeed) + \
                        " - speedBasedOnLastSpeed: " + str(speedBasedOnLastSpeed) + \
                        " - index: " + str(i)
                        currentSpeed = min(currentSpeed, speedBasedOnLastSpeed)
                        nextMessage.waypoints[i].twist.twist.linear.x = currentSpeed

                        lastWp = nextMessage.waypoints[i]

                self.final_waypoints_pub.publish(nextMessage)

            rate.sleep()

    def pose_cb(self, msg):
        self.currentPose = msg.pose

    def waypoints_cb(self, waypoints):
        self.baseWaypoints = waypoints.waypoints

    def traffic_cb(self, msg):
        self.trafficLight = msg.data

    def obstacle_cb(self, msg):
        # Callback for /obstacle_waypoint message. Not needed
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

    def distance2(self, wp1, wp2):
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        dist = dl(wp1.pose.pose.position, wp2.pose.pose.position)
        return dist

    def distanceToWaypoint(self, waypoint):
        x1 = waypoint.pose.pose.position.x
        y1 = waypoint.pose.pose.position.y

        #vehicle coordinates
        xv = self.currentPose.position.x
        yv = self.currentPose.position.y

        return math.sqrt((x1-xv)**2 + (y1-yv)**2)

    '''
    If needed we could make some assumptions and
    start at the last waypoint and only find the first local minimum.

    This would probably decrese the number of waypoints that needs to be searched through.
    '''
    def closestWaypoint(self, waypoints):
        closest = 0
        minDistance = 10**12
        for i, wp in enumerate(waypoints):
            temp = self.distanceToWaypoint(wp)
            if temp < minDistance:
                closest = i
                minDistance = temp

        return closest

    '''
    return: The index of the next waypoint
    '''
    def nextWaypoint(self, waypoints):
        #TODO What to do when we run out of waypoints?
        wp1 = self.closestWaypoint(waypoints)
        wp2 = wp1 + 1
        x1 = waypoints[wp1].pose.pose.position.x
        y1 = waypoints[wp1].pose.pose.position.y
        x2 = waypoints[wp2].pose.pose.position.x
        y2 = waypoints[wp2].pose.pose.position.y

        #vehicle coordinates
        xv = self.currentPose.position.x
        yv = self.currentPose.position.y

        #Changing origin to waypoint1
        x2 -= x1
        y2 -= y1
        xv -= x1
        yv -= x1

        #Dot product
        dot = x2 * xv + y2 * yv

        '''
        If the dot product / scalar product is less than 0 the
        vectors to wp2 and the vehicle points in opposite (kind of) directions.
        In that case we have not yet passed wp1.

        If the scalar product is greter than 0, the vectors have the same direction
        and we have passed wp1, wp2 is the next waypoint.

        In the edge case scaler product = 0, we are currently passing wp1.
        So wp2 is the next waypoint.
        '''
        if dot<0:
            return wp1
        else:
            return wp2

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
