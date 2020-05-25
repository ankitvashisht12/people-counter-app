"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def extract_info_and_draw_boxes(frame, result, args, width, height, prob_threshold):
    '''
    Extract Information and draw bounding boxes onto the frame.
    '''
    person_count = 0
    xmin = None
    xmax = None
    ymin = None
    ymax = None
    for box in result[0][0]: # Output shape is 1x1x100x7
        if int(box[1]) == 1 :
            conf = box[2]
            if conf >= float(prob_threshold):
                person_count += 1
                xmin = int(box[3] * width)
                ymin = int(box[4] * height)
                xmax = int(box[5] * width)
                ymax = int(box[6] * height)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,0,255))
                
    return frame, person_count, ((xmin, ymin), (xmax, ymax))


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def is_prev_bounding_box_in_center_pos(prev_box, width):
    """
    Checks if the previous bounding box lies in center then 
    return True else return false
    """
    
    if( width - 300 < prev_box[0][0]):
        return False
    else:
        return True

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    
    ## Variable Intialization
    input_file = str(args.input)
    model_xml = args.model
    cpu_ext = args.cpu_extension
    device = args.device
    total_count =0
    curr_count = 0
    prev_count = 0
    duration = 0
    frame_buffer = 0
    prev_frame = 0
    prev_box = 0
    flag_mine = True
    
    # Initialise the class
    infer_network = Network()
    
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(model_xml, device, cpu_ext)
    net_input_shape = infer_network.get_input_shape()
    
    ### TODO: Handle the input stream ###
    img_flag = False
    
    if input_file.endswith('.jpeg') or input_file.endswith('.png') or input_file.endswith('.bmp') or input_file.endswith('.jpg'):
        img_flag = True
    elif input_file.upper() == "CAM":
        input_file = 0
  
    cap = cv2.VideoCapture(input_file)
    cap.open(input_file)
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    if not img_flag:
        out = cv2.VideoWriter('out.mp4', 0x00000021, 30, (width, height))
    else:
        out = None
        img_frame = cv2.imread(input_file, cv2.IMREAD_COLOR)
        p_img = cv2.resize(img_frame.copy(), (net_input_shape[3], net_input_shape[2]))
        p_img = p_img.transpose((2,0,1))
        p_img = p_img.reshape(1, *p_img.shape)
        infer_network.exec_net(p_img)
        if infer_network.wait() == 0:
            result = infer_network.get_output()
            p_img, curr_count, box = extract_info_and_draw_boxes(img_frame.copy(), result, args, width, height, prob_threshold)
            cv2.putText(p_img, "People Count : "+str(curr_count), (30, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            cv2.imwrite('output_img.jpg', p_img)
            
        client.disconnect()
        cap.release()
        return
        
        
    ### TODO: Loop until stream is over ###
    while(cap.isOpened()):    
        
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        
        ### TODO: Pre-process the image as needed ###
        p_frame = cv2.resize(frame.copy(), (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        
        ### TODO: Start asynchronous inference for specified request ###
        infer_network.exec_net(p_frame)
        
        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            
            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output()
            p_frame, curr_count, box  = extract_info_and_draw_boxes(frame.copy(), result, args, width, height, prob_threshold)
            
            
            if(box[0][0] is not None):
                prev_box = box
                
            if prev_count > curr_count:
                if frame_buffer <= 10:
                    curr_count = prev_count
                    frame_buffer+=1
                elif is_prev_bounding_box_in_center_pos(prev_box, width):
                    frame_buffer = 0
                    curr_count = prev_count
                else:
                    frame_buffer = 0
                    duration = time.time() - start_time
                    client.publish("person/duration", json.dumps({"duration": duration}))
            elif curr_count > prev_count :
                start_time = time.time()
                frame_buffer = 0
                total_count += curr_count - prev_count
                client.publish("person", json.dumps({"total":total_count}))
    
            
            prev_count = curr_count
            client.publish("person", json.dumps({"count":curr_count}))
            
            ### TODO: Extract any desired stats from the results ###     
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            
        
        ### TODO: Send the frame to the FFMPEG server ###
        out.write(p_frame)
        
        sys.stdout.buffer.write(p_frame)
        sys.stdout.flush()
        
        if key_pressed == 27:
            break
        ### TODO: Write an output image if `single_image_mode` ###
    if not img_flag:
        out.release()
    client.disconnect()
    cap.release()
    return 

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
