# modal runner for the setup notebook -> task 1 on the gpu every reported run used
#
# the notebook itself is the manual's kaggle setup and runs unchanged on a kaggle t4 x2 notebook. this runs it on the
# same 1 x rtx pro 6000 as training, generation and the persona task, so the printed gpu and versions are the ones the
# results came from. it clones the pushed repo from github, so push first. a short run, launched like the others:
#     modal run --detach setup/run_setup_modal.py
#
# pull the results back afterwards (executed notebook, environment.json):
#     modal volume get agaia /pa2/setup/ ./setup/modal_run
#
# watch a detached run:  modal app logs agaia-pa2-setup

from pathlib import Path

import modal

APP_NAME = 'agaia-pa2-setup'
VOLUME_NAME = 'agaia'
VOLUME_PATH = '/mnt/agaia'
REMOTE_TASK_DIR = f'{VOLUME_PATH}/pa2/setup'
NOTEBOOK = Path(__file__).with_name('setup.ipynb')

GPU = 'RTX-PRO-6000' #the same card as every other run
CPU_CORES = 8.0
MEMORY_MB = 32768
TIMEOUT_SECONDS = 20 * 60 #a clone, a pip install and the tests

#only what a kaggle image already ships with -> a cuda torch build and a jupyter kernel. transformers, datasets and
#the rest are left out on purpose, so the notebook's install step does the same work here as it does on kaggle
image = (
    modal.Image.debian_slim(python_version='3.11')
    .apt_install('git')
    #blackwell cards need cuda 12.8 builds, the default pypi wheels are older and would fail to launch a kernel
    .pip_install('torch', index_url='https://download.pytorch.org/whl/cu128')
    .pip_install('papermill==2.7.0', 'ipykernel')
    .add_local_file(NOTEBOOK, remote_path='/root/setup.ipynb', copy=True)
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME, image=image)

#the only change from the manual's kaggle setup is the hardware, and that is a reported deviation
PARAMETERS = {'EXPECTED_GPUS': 1, 'EXPECTED_GPU_NAME': 'RTX PRO 6000'}


@app.function(gpu=GPU, cpu=CPU_CORES, memory=MEMORY_MB, timeout=TIMEOUT_SECONDS, volumes={VOLUME_PATH: volume})
def run():
    import logging

    import papermill

    #papermill forwards each cell's stdout through python logging at info level, this handler shows it in the logs
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    Path(REMOTE_TASK_DIR).mkdir(parents=True, exist_ok=True)
    try:
        papermill.execute_notebook(
            '/root/setup.ipynb', f'{REMOTE_TASK_DIR}/setup_executed.ipynb',
            parameters=PARAMETERS,
            cwd=REMOTE_TASK_DIR, #the clone and ./outputs land here, on the volume
            kernel_name='python3',
            progress_bar=False,
            log_output=True,
        )
    finally:
        #committed even when a cell raises, so the partially executed notebook shows where it stopped
        volume.commit()
        print(f'results committed to {REMOTE_TASK_DIR}')


@app.local_entrypoint()
def main():
    run.remote()
