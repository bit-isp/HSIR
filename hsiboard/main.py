import streamlit as st
import argparse

import viewer
import cmp
import util

page_names_to_funcs = {
    "Viewer": viewer.main,
    "Comparator": cmp.main,
}

parser = argparse.ArgumentParser('HSIR Board')
parser.add_argument('--logdir', default='results')
args = parser.parse_args()


selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())

util.set_page_container_style()
page_names_to_funcs[selected_page](args.logdir)
