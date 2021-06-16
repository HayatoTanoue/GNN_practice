
FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel
RUN  echo "$HTTP_PROXY, $HTTPS_PROXY"
ENV DEBCONF_NOWARNINGS yes
RUN apt-get update -y\
	&& apt-get install -y --no-install-recommends \
	 wget since apt-utils
RUN  apt-get install -y git
RUN conda install -c conda-forge ipywidgets nodejs python-language-server \
    r-languageserver jupyterlab=2.1 ujson=1.35 jedi=0.15.2 parso=0.5.2
RUN pip install --upgrade pip
RUN pip install jupyter-lsp matplotlib networkx "jupyterlab-kite>=2.0.2"

# dawnload torch geometric
RUN pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
RUN pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
RUN pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
RUN pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
RUN pip install torch-geometric
