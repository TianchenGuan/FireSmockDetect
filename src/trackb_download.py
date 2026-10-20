"""Track B: download pyronear/pyro-sdis at pinned revision into data/raw/pyro_sdis."""
from huggingface_hub import snapshot_download

REV = "a1e553ec4d806f71fc6db744cc22bc3469487382"
path = snapshot_download(
    repo_id="pyronear/pyro-sdis",
    repo_type="dataset",
    revision=REV,
    local_dir="/local/home/tianrgua/cvproject/data/raw/pyro_sdis",
    max_workers=16,
)
print("DONE", path)
