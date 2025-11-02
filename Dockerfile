FROM tensorflow/tensorflow:2.16.1
RUN mkdir -p /program
WORKDIR /program

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt /program/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


# Install JupyterLab
RUN pip install jupyterlab


# Expose Jupyter port
EXPOSE 8888

# Run JupyterLab when container starts
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]