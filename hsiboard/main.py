import argparse
import streamlit as st
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import viewer
import util
import cmp

def main(logdir):
    page_names_to_funcs = {
        "Viewer": viewer.main,
        "Comparator": cmp.main,
    }

    selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())

    util.set_page_container_style()
    page_names_to_funcs[selected_page](logdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('HSIR Board')
    parser.add_argument('--logdir', default='results')
    args = parser.parse_args()

    main(args.logdir)
