#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ===============================================================================
#
# Copyright (c) 2026 Hai Liang Wang<hailiang.hl.wang@gmail.com> All Rights Reserved
#
#
# File: /media/vision/git/touchtext/test.py
# Author: Hai Liang Wang
# Date: 2026-03-06:16:10:07
#
# ===============================================================================

"""
This script does not support python2.
"""
__copyright__ = "Copyright (c) 2026 Hai Liang Wang<hailiang.hl.wang@gmail.com> All Rights Reserved"
__author__ = "Hai Liang Wang"
__date__ = "2026-03-06:16:10:07"

import os, sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, curdir)


from touchtext.datasets import Multi30k

def main():
    train_iter, valid_iter, test_iter = Multi30k(language_pair=("de", "en"))
    print("train_iter[0]", train_iter[0])

if __name__ == '__main__':
    main()
