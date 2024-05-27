import os
import re
import subprocess

SCRIPT_ROOT = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "scripts", "classification"
)
skip_list = [
    # "Autoformer",
    # "Crossformer"
    # "FEDformer",
    # "Informer",
    # "iTransformer",
    # "MTST",
    # "Nonstationary",
    # "PatchTST",
    # "Reformer",
    # "Transformer",
    # "Medformer",
]

for dir_path, dir_names, file_names in os.walk(SCRIPT_ROOT):
    file_names = sorted(
        filter(
            lambda x: x.endswith(".sh") and x[:-3].split("_")[0] not in skip_list,
            file_names,
        ),
        key=lambda x: x.split("_")[::-1],
    )
    for file_name in file_names:
        script_path = os.path.join(dir_path, file_name)
        print(f"Running {script_path}")
        subprocess.run(["bash", script_path], check=True)
