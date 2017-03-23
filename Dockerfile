FROM jupyter/notebook
MAINTAINER gatakaba

RUN apt-get update && apt-get install -y wget
# install packages
RUN pip3 install --upgrade pip
RUN pip install numpy
RUN pip install scipy
RUN pip install sympy
RUN pip install pandas
RUN pip install matplotlib

RUN pip install sklearn
RUN pip install seaborn
RUN pip install ipywidgets


WORKDIR /root
RUN pip install git+https://github.com/gatakaba/pySDNN.git@base_network
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension
