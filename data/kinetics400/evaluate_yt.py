#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: shiv
"""

import json
import os
import time
from pytube import YouTube

import constants as C
import rgb_flow
import evaluate_i3d

# Constants
TRAIN_JSON_DATA = '/home/shiv/kinetics-i3d/data/kinetics400/train00.json'
DOWNLOAD_DIR = '/tmp/yt/'
RESULTS_LOG_FILE = '/home/shiv/kinetics-i3d/results.txt'

def evaluate_json_sample():
    num_success = 0
    num_entries = 0
    results_log_file = open(RESULTS_LOG_FILE, 'w+')
    crgb_flow = rgb_flow.ComputeRGBFlow()
    with open(TRAIN_JSON_DATA) as json_file:
        data = json.load(json_file)
        for key in data:
            new_item = data[key]
            annot = new_item['annotations']
            #duration = new_item['duration']
            url = new_item['url']
            try:
                yt = YouTube(url)
            except:
                print("Connection error")
                results_log_file.write('Connection error')
                continue

            mp4files = yt.streams.filter(res='360p', file_extension='mp4',progressive=True)
            stream = mp4files.first()
            stream.download(DOWNLOAD_DIR)
            
            filename = DOWNLOAD_DIR+stream.default_filename
            seg_start_time_sec = annot['segment'][0]
            seg_end_time_sec = annot['segment'][1]
            label = annot['label']
            #fps = stream.fps
            
            start_time = time.time()
            
            print('Extract RGB & Flow...', filename)
            results_log_file.write('%s\n' % stream.default_filename)
            num_frames = crgb_flow.compute_rgb_flow(filename, DOWNLOAD_DIR, seg_start_time_sec, seg_end_time_sec)
            result = evaluate_i3d.evaluate(num_frames, DOWNLOAD_DIR+C._OUTPUT_RGB_NPY, DOWNLOAD_DIR+C._OUTPUT_FLOW_NPY, label, results_log_file)
            if (result == 1):
                num_success = num_success + 1
            num_entries = num_entries + 1
            print('Success: %d %d ' % (num_success, num_entries))
            print('Extract rgb & flow in sec: ', time.time() - start_time)  
            os.remove(filename) # save space
    results_log_file.write('Success: %d %d\n' % (num_success, num_entries))
    results_log_file.close()
    
def main():
    evaluate_json_sample()

if __name__ == '__main__':
    main()
