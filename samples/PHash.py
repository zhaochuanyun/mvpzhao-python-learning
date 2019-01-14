#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
https://blog.csdn.net/m_buddy/article/details/78887248
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import urllib


# extract feature
# lines: src_img path
def extra_featrue(lines, new_rows=64, new_cols=64):
    for name in lines:
        print(name)
        ori_img = Image.open(name.strip())
        feature_img = ori_img.resize((new_rows, new_cols))
        feature_img = feature_img.convert('L')
        mean_value = np.mean(feature_img)
        feature = feature_img >= mean_value
        matrix_feature = np.matrix(feature, np.int8)
        if 'features' in locals():
            temp = np.reshape(matrix_feature, (1, new_cols * new_rows))
            features = np.vstack([matrix_feature, temp])
        else:
            features = np.matrix(np.reshape(matrix_feature, (1, new_cols * new_rows)))
    return features


extra_featrue(['characters/0.png'])