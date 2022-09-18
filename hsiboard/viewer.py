import argparse
from os.path import join
from hsiboard.box import addbox

from util import *

def main(logdir):

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

        nrow = st.slider('number of rows', min_value=3, max_value=50, value=9)
        ncol = st.slider('number of columns', min_value=1, max_value=20, value=6)

        enable_enlarge = st.checkbox('Enlarge')
        crow = st.slider('row coordinate', min_value=0.0, max_value=1.0, value=0.2)
        ccol = st.slider('col coordinate', min_value=0.0, max_value=1.0, value=0.2)

    grids = make_grid(nrow, ncol)
    imgs = load_imgs(join(logdir, selected_method, selected_dataset, selected_vis_type))
    details = load_per_image_stat(join(logdir, selected_method, selected_dataset))
    with st.container():
        idx = 0
        for name, img in imgs.items():
            name = os.path.splitext(name)[-2]
            if enable_enlarge:
                h, w = img.shape[:2]
                img = addbox(img.copy(), (int(h*crow), int(w*ccol)))
            ct = grids[idx // ncol][idx % ncol]
            ct.image(img, caption='%s [%.4f]'% (name, details[name]['MPSNR']))
            idx += 1

    # stat = load_stat(logdir)
    # st.header(selected_method)
    # st.table(stat[selected_method])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('HSIR Board')
    parser.add_argument('--logdir', default='results')
    args = parser.parse_args()

    set_page_container_style(max_width_100_percent=True)
    main(args.logdir)
