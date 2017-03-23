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
RUN wget https://raw.githubusercontent.com/gatakaba/pySDNN/develop/notebook/SDNN.ipynb
RUN wget https://raw.githubusercontent.com/gatakaba/pySDNN/develop/notebook/PP-P.ipynb
RUN wget https://raw.githubusercontent.com/gatakaba/pySDNN/develop/notebook/PP-A.ipynb

RUN pip install git+https://github.com/gatakaba/pySDNN.git@develop
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension

