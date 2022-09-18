import argparse
import imageio
from os.path import join
import numpy as np
import json

from util import *
from box import *


def main(logdir):

    with st.sidebar:
        st.title('HSIR Board')
        st.subheader('Comparator')
        st.caption(f'Directory: {logdir}')
        methods = listdir(logdir)
        selected_methods = st.sidebar.multiselect(
            "Select Method",
            methods,
            default=methods
        )
        datasets_list = [listdir(join(logdir, m)) for m in selected_methods]
        datasets = set(datasets_list[0]).intersection(*map(set, datasets_list))
        selected_dataset = st.sidebar.selectbox(
            "Select Dataset",
            datasets
        )
        selected_vis_type = st.sidebar.selectbox(
            "Select Image Type",
            ['color', 'gray']
        )

        img_names = os.listdir(join(logdir, selected_methods[0], selected_dataset, selected_vis_type))
        selected_img = st.sidebar.selectbox(
            "Select Image",
            img_names
        )

        enable_enlarge = st.checkbox('Enlarge')
        enable_diff = st.checkbox('Difference Map')
        enable_sidebyside = st.checkbox('Side by Side')
        
        st.header('Box')
        crow = st.slider('row coordinate', min_value=0.0, max_value=1.0, value=0.2)
        ccol = st.slider('col coordinate', min_value=0.0, max_value=1.0, value=0.2)
        vmax = st.slider('vmax', min_value=0.0, max_value=1.0, value=0.1)

        st.header('Layout')
        nrow = st.slider('number of rows', min_value=3, max_value=50, value=9)
        ncol = st.slider('number of columns', min_value=1, max_value=20, value=4)

    grids = make_grid(nrow, ncol)
    imgs = {}
    stats = {}
    for m in selected_methods:
        img = imageio.imread(join(logdir, m, selected_dataset, selected_vis_type, selected_img))
        imgs[m] = np.array(img, dtype=np.float32) / 255
        stat = json.load(open(join(logdir, m, selected_dataset, 'log.json')))
        stats[m] = stat['detail'][os.path.splitext(selected_img)[-2]]

    gt = imgs['gt']
    
    with st.container():
        idx = 0
        for name, im in imgs.items():
            img = im.copy()
            name = os.path.splitext(name)[-2]
            if enable_diff:
                diff = np.abs(img-gt)
                if len(diff.shape) == 3: diff = diff.mean(-1)
                img = convert_color(diff, vmax=vmax)

            if enable_sidebyside:
                h, w = img.shape[:2]
                img = addbox_with_diff(img, gt, (int(h * crow), int(w * ccol)), vmax=vmax)

            elif enable_enlarge:
                h, w = img.shape[:2]
                img = addbox(img, (int(h * crow), int(w * ccol)))
                
                
            ct = grids[idx // ncol][idx % ncol]
            ct.image(img, caption='%s [%.4f]' % (name, stats[name]['MPSNR']), clamp=[0,1])
            idx += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser('HSIR Board')
    parser.add_argument('--logdir', default='results')
    args = parser.parse_args()

    set_page_container_style()
    main(args.logdir)
