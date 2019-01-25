#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 12:28:20 2019

@author: xulin
"""

import binvox_rw

# Read the voxel data from a .binvox file.
def read_voxel_data(model_path):
    with open(model_path, 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)
        return model.data
    
def write_binvox_file(pred, filename):
    with open(filename, 'wb') as f:
        voxel = binvox_rw.Voxels(pred, [32, 32, 32], [0, 0, 0], 1, 'xzy')
        binvox_rw.write(voxel, f)
        f.close()