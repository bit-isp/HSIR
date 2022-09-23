import argparse
from util import *


def main(logdir):

    with st.sidebar:
        st.title('HSIR Board')
        st.subheader('Table')
        st.caption(f'Directory: {logdir}')
        methods = listdir(logdir, exclude=['gt'])
        selected_methods = st.sidebar.multiselect(
            "Select Method",
            methods,
            default=methods,
        )

    stat = load_stat(logdir)
    
    st.subheader('Not loaded')
    st.text(set(selected_methods)-set(stat.keys()))
    
    if len(selected_methods) == 1:
        selected_method = selected_methods[0]
        print(stat[selected_method][0])
        st.header(selected_method)
        st.dataframe(stat[selected_method])
    else:
        table = {}
        for m in stat.keys():
            for d in stat[m]:
                row = {'Method': m}
                for k, v in d.items():
                    if k != 'Name':
                        row[k] = v
                if d['Name'] not in table:
                    table[d['Name']] = []
                table[d['Name']].append(row)
        for k, v in table.items():
            st.subheader(k)
            st.dataframe(v)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('HSIR Board')
    parser.add_argument('--logdir', default='results')
    args = parser.parse_args()

    main(args.logdir)
