From pytorch/pytorch:latest
LABEL maintainer="theabc50111@gmail.com"

COPY ["init.sh", "summary.py", "__init__.py", "requirements.txt", "tracker.jupyterlab-settings", "ipython_kernel_config.py", "jupyter_lab_config.py", "themes.jupyterlab-settings", "/root/"]
RUN ["/bin/bash", "-c", "/root/init.sh"]
WORKDIR /workspace

CMD jupyter lab
