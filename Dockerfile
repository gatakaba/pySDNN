FROM jupyter/notebook
MAINTAINER gatakaba

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

RUN pip install git+https://github.com/gatakaba/pySDNN.git
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension
