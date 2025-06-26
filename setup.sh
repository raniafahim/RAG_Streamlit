cd ~/work/Decoupage/ 
uv venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uv pip install notebook ipykernel
python -m ipykernel install --user --name=my-uv-env --display-name "Python (uv)"