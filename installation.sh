

cd ~/Documents/GitHub/sturdy-barnacle

# Create virtual environment
python3.11 -m venv myenv
source myenv/bin/activate


pip install numpy==1.23.5
pip install matplotlib
pip install jupyterlab


python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

pip install torch torchvision torchaudio

python -m ipykernel install --user --name=venv

pip install ipywidgets widgetsnbextension pandas-profiling opencv-python


pip install "numpy<2" --force-reinstall
