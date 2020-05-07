#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
file_utility.py

Useful functions that manipulate file.
"""

import os
import cv2

def createFolder(directory):
    """Create folder in the specified path unless it already exists.
    
    Parameters
    ----------
    directory : str
        The file directory to be created in the specified path
    Examples
    --------
    createFolder('./pics/')
    """
    try:
        # Determine whether the directory exists.
        if not os.path.exists(directory):
            # Try to create the directory.
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
        
def makeVideo(pics_folder,video_folder,video_name,video_fps,T):
    """Create video using the pictures stored in the specified folder.
    
    Parameters
    ----------
    pics_folder : str
        The file location of the stored pictures
    video_folder: str
        The file location of the stored videos
    video_name : str
        The file name of the video created
    video_fps: float
        Frames per second
    T: int
        The total time steps used to create the video
    """
    img_array = []
    for i in range(T):
        img = cv2.imread(pics_folder+"{}.png".format(i))
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    
    out = cv2.VideoWriter(video_folder+video_name+'.mp4',cv2.VideoWriter_fourcc(*'FMP4'), video_fps, size)
     
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()