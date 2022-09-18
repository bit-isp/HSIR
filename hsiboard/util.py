import json
import os
from collections import OrderedDict
from os.path import exists, join

import imageio
import numpy as np
import streamlit as st

BACKGROUND_COLOR = 'white'
COLOR = 'black'


def set_page_container_style(
    max_width: int = 1100, max_width_100_percent: bool = False,
    padding_top: int = 4, padding_right: int = 2, padding_left: int = 2, padding_bottom: int = 10,
    color: str = COLOR, background_color: str = BACKGROUND_COLOR,
):
    if max_width_100_percent:
        max_width_str = f'max-width: 100%;'
    else:
        max_width_str = f'max-width: {max_width}px;'
    st.markdown(
        f'''
            <style>
                .reportview-container .sidebar-content {{
                    padding-top: {padding_top}rem;
                }}
                .appview-container .main .block-container {{
                    {max_width_str}
                    padding-top: {padding_top}rem;
                    padding-right: {padding_right}rem;
                    padding-left: {padding_left}rem;
                    padding-bottom: {padding_bottom}rem;
                }}
                .reportview-container .main {{
                    color: {color};
                    background-color: {background_color};
                }}
            </style>
            ''',
        unsafe_allow_html=True,
    )


def listdir(path, exclude=[]):
    outputs = []
    for folder in os.listdir(path):
        if os.path.isdir(join(path, folder)) and folder not in exclude:
            outputs.append(folder)
    return outputs


def load_method_stat(logdir):
    stat = []
    for folder in listdir(logdir):
        path = join(logdir, folder, 'log.json')
        if exists(path):
            data = json.load(open(path, 'r'))
            s = OrderedDict()
            s['Name'] = folder
            s.update(data['avg'])
            stat.append(s)
    return stat


def load_stat(logdir):
    total_stat = {}
    for folder in listdir(logdir, exclude=['gt']):
        stat = load_method_stat(join(logdir, folder))
        total_stat[folder] = stat
    return total_stat

def load_per_image_stat(logdir):
    path = join(logdir, 'log.json')
    data = json.load(open(path, 'r'))
    return data['detail']

def load_imgs(path):
    out = {}
    for name in os.listdir(path):
        if name.endswith('png'):
            img = imageio.imread(join(path, name))
            img = np.array(img).clip(0, 255)
            out[name] = img
    return out


def make_grid(rows, cols):
    grid = [0] * rows
    for i in range(rows):
        with st.container():
            grid[i] = st.columns(cols)
    return grid
