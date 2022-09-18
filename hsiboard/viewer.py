import argparse
from os.path import join

from util import *
from box import *


def main(logdir):
    set_page_container_style()
    with st.sidebar:
        st.title('HSIR Board')
        st.subheader('Viewer')
        st.caption(f'Directory: {logdir}')
        methods = listdir(logdir, exclude=['gt'])
        selected_method = st.sidebar.selectbox(
            "Select Method",
            methods
        )
        datasets = listdir(join(logdir, selected_method))
        selected_dataset = st.sidebar.selectbox(
            "Select Dataset",
            datasets
        )
        selected_vis_type = st.sidebar.selectbox(
            "Select Image Type",
            ['color', 'gray']
        )

        st.header('Box')
        enable_enlarge = st.checkbox('Enlarge')
        crow = st.slider('row coordinate', min_value=0.0, max_value=1.0, value=0.2)
        ccol = st.slider('col coordinate', min_value=0.0, max_value=1.0, value=0.2)
        selected_box_pos = st.sidebar.selectbox(
            "Select Box Position",
            ['Bottom Right', 'Bottom Left', 'Top Right', 'Top Left'],
        )

        st.header('Layout')
        ncol = st.slider('number of columns', min_value=1, max_value=20, value=6)

    imgs = load_imgs(join(logdir, selected_method, selected_dataset, selected_vis_type))

    nrow = len(imgs) // ncol + 1
    grids = make_grid(nrow, ncol)
    details = load_per_image_stat(join(logdir, selected_method, selected_dataset))
    with st.container():
        idx = 0
        for name, img in imgs.items():
            name = os.path.splitext(name)[-2]
            if enable_enlarge:
                h, w = img.shape[:2]
                img = addbox(img.copy(), (int(h * crow), int(w * ccol)),
                             bbpos=mapbbpox[selected_box_pos])
            ct = grids[idx // ncol][idx % ncol]
            ct.image(img, caption='%s [%.4f]' % (name, details[name]['MPSNR']))
            idx += 1



if __name__ == '__main__':
    parser = argparse.ArgumentParser('HSIR Board')
    parser.add_argument('--logdir', default='results')
    args = parser.parse_args()

    main(args.logdir)
