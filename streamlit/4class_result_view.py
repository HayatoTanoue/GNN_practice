import streamlit as st

from glob import glob

st.title("GCN result viewer")

select_data = st.radio("select dataset", ["4 class origin data", "bench mark dataset"])

if select_data == "4 class origin data":
    num_node = st.radio("select number of node", ["100", "1000"])
    re_dir = sorted(glob("../4class_result/{}/*".format(num_node)))

    for path in re_dir:
        st.subheader(path.split("/")[-1])
        col1, col2 = st.beta_columns(2)
        col1.image(
            path + "/learning.png", caption="learning curve", use_column_width=True
        )
        col2.image(
            path + "/conf_matrix.png", caption="conf matrix", use_column_width=True
        )

else:
    re_dir = sorted(glob("../notebooks/data/TUDataset/*"))
    for path in re_dir:
        st.subheader(path.split("/")[-1])
        col1, col2 = st.beta_columns(2)
        col1.image(
            path + "/result/learning.png",
            caption="learning curve",
            use_column_width=True,
        )
        col2.image(
            path + "/result/conf_matrix.png",
            caption="conf matrix",
            use_column_width=True,
        )