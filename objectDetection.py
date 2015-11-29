# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 17:35:34 2015

@author: 4chennur
"""

#!/usr/bin/env python

""" cv_bridge_demo.py - Version 0.1 2011-05-29

    A ROS-to-OpenCV node that uses cv_bridge to map a ROS image topic and optionally a ROS
    depth image topic to the equivalent OpenCV image stream(s).
    
    Created for the Pi Robot Project: http://www.pirobot.org
    Copyright (c) 2011 Patrick Goebel.  All rights reserved.

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.
    
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details at:
    
    http://www.gnu.org/licenses/gpl.html
      
"""

import roslib; #roslib.load_manifest('rbx_vision')
import rospy
import sys
import cv2
import cv
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

depth_display_image = None

class objectDetection():
    def __init__(self):
        self.nodeName = "object_detection"
        
        rospy.init_node(self.nodeName)
        
        # What we do during shutdown
        rospy.on_shutdown(self.cleanup)
        
        # Create the OpenCV display window for the RGB image
        self.cv_window_name = self.nodeName
        cv.NamedWindow(self.cv_window_name, cv.CV_WINDOW_NORMAL)
        cv.MoveWindow(self.cv_window_name, 25, 75)
        
        # And one for the depth image
        cv.NamedWindow("Depth Image", cv.CV_WINDOW_NORMAL)
        cv.MoveWindow("Depth Image", 25, 350)
        
        '''Initialize ros publisher'''
        # topic where we publish
        self.imagePub = rospy.Publisher("/output/image_raw/compressed",
            Image, queue_size = 10)
        # Create the cv_bridge object
        self.bridge = CvBridge()
        
        # Subscribe to the camera image and depth topics and set
        # the appropriate callbacks
        self.imageSub = rospy.Subscriber("/camera/rgb/image_rect_color",
                            Image, self.imageCallback, queue_size=1)
        self.depthSub = rospy.Subscriber("/camera/depth/image_rect",
                            Image, self.depthCallback, queue_size=1)
        
        rospy.loginfo("Waiting for image topics...")

    def imageCallback(self, rosImage):
        # Use cv_bridge() to convert the ROS image to OpenCV format
        try:
            frame = self.bridge.imgmsg_to_cv2(rosImage, "bgr8")
        except CvBridgeError, e:
            print e
        
        # Convert the image to a Numpy array since most cv2 functions
        # require Numpy arrays.
        frame = np.array(frame, dtype=np.uint8)
        
        # Process the frame using the process_image() function
        displayImage = self.processImage(frame)
                       
        # Display the image.
        cv2.imshow(self.nodeName, displayImage)
        
        # Process any keyboard commands
        self.keystroke = cv.WaitKey(5)
        if 32 <= self.keystroke and self.keystroke < 128:
            cc = chr(self.keystroke).lower()
            if cc == 'q':
                # The user has press the q key, so exit
                rospy.signal_shutdown("User hit q key to quit.")
        
                
    def depthCallback(self, ros_image):
        global depthDisplayImage
        # Use cv_bridge() to convert the ROS image to OpenCV format
        try:
            # The depth image is a single-channel float32 image
            depthImage = self.bridge.imgmsg_to_cv2(rosImage, "16UC1")
        except CvBridgeError, e:
            print e

        # Convert the depth image to a Numpy array since most cv2 functions
        # require Numpy arrays.
        depthArray = np.array(depthImage, dtype=np.float32)
                
        # Normalize the depth image to fall between 0 (black) and 1 (white)
        depthDisplayImage = cv2.normalize(depthArray, depthArray, 0, 1, cv2.NORM_MINMAX)
        
        # Process the depth image
        #depth_display_image = self.process_depth_image(depth_array)
        depthDisplayImage = depthDisplayImage[:,:,0];
        # Display the result
        cv2.imshow("Depth Image", depthDisplayImage)
#        print 'Hello'
        depthValue =  self.getDepth(depthDisplayImage,20,3)
        print depthValue
#        print 'Hello2'
          
    def processImage(self, frame):
        # Convert to greyscale
        grey = cv2.cvtColor(frame, cv.CV_BGR2GRAY)
        
        # Blur the image
        grey = cv2.blur(grey, (7, 7))
        
        # Compute edges using the Canny edge filter
        edges = cv2.Canny(grey, 15.0, 30.0)
        
        return edges
    
    def processDepthImage(self, frame):
        # Just return the raw image for this demo
        return frame
    
    def cleanup(self):
        print "Shutting down vision node."
        cv2.destroyAllWindows()  
        
            ### Create CompressedIamge ####
    def getDepth(self,image,x,y):
        print 'Hi'
        try:
            return image[y][x] 
        except:
            print "Error occured in getDepth...\n"
            return -1
        
    
def main(args):       
    try:
        objectDetection()
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down vision node."
        cv.DestroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
    
    