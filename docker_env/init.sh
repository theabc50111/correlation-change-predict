#!/bin/bash
echo "start to initualization!!!!!!!"

# install common python pacakge
pip install -r /root/requirements.txt

# setting ip, password, allow-root of jupyter lab
mkdir -p ~/.jupyter
cp /root/jupyter_lab_config.py ~/.jupyter/ 

# setting autoload, autotime to ipython
ipython profile create
cp /root/ipython_kernel_config.py ~/.ipython/profile_default/


# Adjusted font-size & theme-color
mkdir -p ~/.jupyter/lab/user-settings/\@jupyterlab/apputils-extension
mv /root/themes.jupyterlab-settings ~/.jupyter/lab/user-settings/\@jupyterlab/apputils-extension/
 
# Adjusted codeCellconfig, e.g. lineNumber, LineWrapper
mkdir -p ~/.jupyter/lab/user-settings/\@jupyterlab/notebook-extension/
mv /root/tracker.jupyterlab-settings ~/.jupyter/lab/user-settings/@jupyterlab/notebook-extension/

# install pytorch-side-package: PYG
TORCH=$(python -c "import torch; print(torch.__version__)")
CUDA=$(python -c "import torch; print(f'cu{torch.version.cuda.replace(\".\", \"\")}')")
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric

# add torch_geometric summary()
cp /root/summary.py /opt/conda/lib/python3\.10/site-packages/torch_geometric/nn
#cp /root/__init__.py /opt/conda/lib/python3\.10/site-packages/torch_geometric/nn

# install Nike-Inc/timeseries-generator.git
apt update && apt install -y git
cd /root
git clone https://github.com/Nike-Inc/timeseries-generator.git
cd /root/timeseries-generator
python setup.py install
cd /opt/conda/lib/python3.10/site-packages/timeseries_generator-0.1.0-py3.10.egg/timeseries_generator
sed -i '5s/from matplotlib.axes._subplots import SubplotBase/from matplotlib.axes import SubplotBase/' base_factor.py
sed -i '138ixxx' holiday_factor.py
sed -i '138s/xxx/            country_name = country_name.replace("_", "")/g' holiday_factor.py
sed -i '97ixxx' generator.py
sed -i '97s/xxx/            if "country" in df.columns:\n                df.loc[::, "country"] = df.loc[::, "country"].apply(lambda x:x.lower().replace("_", "").replace(" ", ""))/g' generator.py
