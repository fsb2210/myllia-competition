import requests
import json
from pathlib import Path
from tqdm.auto import tqdm

BASE_URL = "https://api.figshare.com/v2"
ITEM_ID = "20029387"
NAME = "rpe1_raw_singlecell_01.h5ad"

r = requests.get(BASE_URL + "/articles/" + str(ITEM_ID))
# Load the metadata as JSON
if r.status_code != 200:
    print('Something is wrong:', r.content)
else:
    metadata = json.loads(r.text)

for file in metadata["files"]:
    if file["name"] == NAME:
        download_url = file["download_url"]
        size = file["size"]

        response = requests.get(download_url, stream=True)
response.raise_for_status()

with open(NAME, "wb") as f, tqdm(total=size, unit="B", unit_scale=True, desc="Downloading") as bar:
    for chunk in response.iter_content(chunk_size=8192):
        if not chunk: break
        f.write(chunk)
        bar.update(len(chunk))

NAME = "jurkat_raw_singlecell_01.h5ad"
download_url = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE264667&format=file&file=GSE264667%5Fjurkat%5Fraw%5Fsinglecell%5F01%2Eh5ad"

response = requests.get(download_url, stream=True)
response.raise_for_status()

total = int(response.headers.get("content-length", 0))
with open(NAME, "wb") as f, tqdm(total=size, unit="B", unit_scale=True, desc="Downloading") as bar:
    for chunk in response.iter_content(chunk_size=8192):
        if not chunk: break
        f.write(chunk)
        bar.update(len(chunk))
