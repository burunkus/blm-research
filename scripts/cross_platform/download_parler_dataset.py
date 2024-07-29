#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 11:29:26 2022

@author: ebukaokpala
"""

import json
import datetime
import sys
import subprocess
import os
import logging
import logging.handlers


def get_filename():
    #ct = datetime.datetime.now()
    #log_name = f"{ct.year}-{ct.month:02d}-{ct.day:02d}_{ct.hour:02d}:{ct.minute:02d}:{ct.second:02d}"
    current_file_name = os.path.basename(__file__).split('.')[0]
    log_name = current_file_name
    return log_name


def get_logger(log_folder,log_filename):
    if os.path.exists(log_folder) == False:
        os.makedirs(log_folder)

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s]:  %(message)s",
        datefmt="%m-%d-%Y %H:%M:%S",
        handlers=[logging.FileHandler(os.path.join(log_folder, log_filename+'.log'), mode='w'),
        logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    return logger


def runcmd(cmd, verbose = False, *args, **kwargs):
    """Adapted from https://www.scrapingbee.com/blog/python-wget/"""
    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True
    )
    std_out, std_err = process.communicate()
    if verbose:
        logging.info(std_out.strip(), std_err)
    pass


def download_dataset(download_url, save_path):
    wget_command = f"wget --directory-prefix={save_path} {download_url}"
    runcmd(wget_command, verbose=True)


def main():
    save_path = "../../../../../zfs/socbd/eokpala/blm_research/data/"
    download_url = "https://zenodo.org/records/4442460/files/parler_data.zip"
    download_dataset(download_url, save_path)
    
    
if __name__ == "__main__":
    log_dir ='./log_folder'
    _ = get_logger(log_dir, get_filename())
    main()
