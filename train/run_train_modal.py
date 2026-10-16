# modal runner for the training notebook -> sft, then the hh-rlhf and ultrafeedback dpo runs
#
# launch it detached, the same way the atml runs were. the container keeps running on modal's side and the results
# land in the volume, so a dropped laptop connection or a closed terminal cannot kill the run:
#     modal run --detach train/run_train_modal.py
#
# the notebook's own constants are the manual's guide configuration for kaggle t4 x2, and that is the default here
# too. --scale swaps in the batch layout described in SCALED_PARAMETERS below, which is what a 96gb card is for:
#     modal run --detach train/run_train_modal.py --scale
#
# pull the results back afterwards (executed notebook, csvs, pngs, checkpoints):
#     modal volume get agaia /pa2/train/ ./train/modal_run
#
# watch a detached run:  modal app logs agaia-pa2-train

from pathlib import Path

import modal

APP_NAME = 'agaia-pa2-train'
VOLUME_NAME = 'agaia' #one volume for this course, everything for this assignment lives under /pa2
VOLUME_PATH = '/mnt/agaia'
#the notebook runs with this as its working directory, so its relative './data' and './outputs' paths land on the
#volume. that is what makes the downloaded model, the datasets and every checkpoint survive between runs
REMOTE_TASK_DIR = f'{VOLUME_PATH}/pa2/train'
NOTEBOOK = Path(__file__).with_name('train.ipynb')
PACKAGE_DIR = Path(__file__).parent.parent / 'alignment' #the from scratch sft/dpo code the notebook imports

GPU = 'RTX-PRO-6000' #a single fixed gpu type, so the hardware is identical across every run and reportable
CPU_CORES = 8.0
MEMORY_MB = 32768
TIMEOUT_SECONDS = 3 * 60 * 60 #three hours, generous even for the unscaled batch 2 layout

image = (
    modal.Image.debian_slim(python_version='3.11') #the python version the atml runs used
    #blackwell cards need cuda 12.8 builds, the default pypi wheels are older and would fail to launch a kernel.
    #this call uses the pytorch index, so it is kept separate from the pypi packages below
    .pip_install('torch', index_url='https://download.pytorch.org/whl/cu128')
    .pip_install(
        'papermill==2.7.0', 'ipykernel', 'transformers>=4.43.0,<5', 'datasets>=2.18,<5', 'huggingface_hub',
        'accelerate', 'safetensors', 'sentencepiece', 'pandas', 'matplotlib', 'tqdm'
    )
    #the notebook's kernel starts in the volume directory, /root on the path is what lets it import alignment
    .env({'PYTHONPATH': '/root'})
    #copy=True bakes the code into the image, so a detached run does not depend on the local machine staying up
    .add_local_dir(PACKAGE_DIR, remote_path='/root/alignment', copy=True)
    .add_local_file(NOTEBOOK, remote_path='/root/train.ipynb', copy=True)
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME, image=image)

#empty means the notebook keeps its own constants, which are the manual's guide values
SPEC_PARAMETERS = {}

#papermill injects these over the notebook's constants cell. only the batch layout and the precision change:
#the effective batch stays at the guide's 32 for both stages, so the number of optimizer updates is unchanged and
#the learning rates are deliberately left alone. there are no lost updates here for a higher rate to make up for
SCALED_PARAMETERS = {
    'SFT_BATCH_SIZE': 32, #2 x 16 accumulation -> 32 x 1, one real batch of 32
    'SFT_GRAD_ACCUM': 1,
    'DPO_BATCH_SIZE': 16, #1 x 32 accumulation -> 16 pairs x 2
    'DPO_GRAD_ACCUM': 2,
    'EVAL_BATCH_SIZE': 32, #evaluation is forward only, so this changes speed and nothing else
    'PRECISION': 'bf16', #blackwell supports bf16 natively and it needs no gradient scaling, fp16 was a t4 choice
    'REFERENCE_HALF_PRECISION': False, #with 96gb the reference can stay fp32, identical to the policy at step 0
}


@app.function(gpu=GPU, cpu=CPU_CORES, memory=MEMORY_MB, timeout=TIMEOUT_SECONDS, volumes={VOLUME_PATH: volume})
def run(parameters):
    import logging

    import papermill

    #papermill forwards each cell's stdout through python logging at info level. without a configured handler only
    #stderr (warnings, progress bars) reaches the modal logs and the training prints are silently dropped
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    volume.reload() #picks up the cached model and datasets, and any run an earlier launch already finished
    Path(REMOTE_TASK_DIR).mkdir(parents=True, exist_ok=True)

    executed = f'{REMOTE_TASK_DIR}/train_executed.ipynb'
    print(f'running train with parameters {parameters or "the notebook defaults"}')

    try:
        papermill.execute_notebook(
            '/root/train.ipynb', executed,
            parameters=parameters,
            cwd=REMOTE_TASK_DIR, #relative paths inside the notebook resolve here, on the volume
            kernel_name='python3',
            progress_bar=False, #the logs are captured, a progress bar would only add noise to them
            log_output=True, #every cell's stdout goes to the modal logs, so a detached run is still followable
        )
    finally:
        #committed even when a cell raises, so the partially executed notebook and whatever results exist are kept
        volume.commit()
        print(f'results committed to {REMOTE_TASK_DIR}')


@app.local_entrypoint()
def main(scale: bool = False):
    run.remote(SCALED_PARAMETERS if scale else SPEC_PARAMETERS)
