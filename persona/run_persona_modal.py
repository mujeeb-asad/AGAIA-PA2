# modal runner for the persona notebook -> trl dpo + lora on smollm2-360m-instruct, then base vs lora generations
#
# launch it detached, the same way the training run was. it needs nothing from the other runs, so it can run at the
# same time as any of them:
#     modal run --detach persona/run_persona_modal.py
#
# --penalty also runs the repetition penalty ablation after the greedy pass, in the same container, with its answers,
# selection and pair files in their own folder. finished lora runs are reused, so this only generates:
#     modal run --detach persona/run_persona_modal.py --penalty
#
# pull the results back afterwards (executed notebook, pilot csv, plots, generations, pair file):
#     modal volume get agaia /pa2/persona/ ./persona/modal_run
#
# watch a detached run:  modal app logs agaia-pa2-persona

from pathlib import Path

import modal

APP_NAME = 'agaia-pa2-persona'
VOLUME_NAME = 'agaia'
VOLUME_PATH = '/mnt/agaia'
#the notebook runs with this as its working directory, so its relative './data' and './outputs' land on the volume
REMOTE_TASK_DIR = f'{VOLUME_PATH}/pa2/persona'
NOTEBOOK = Path(__file__).with_name('persona.ipynb')
REPO_DIR = Path(__file__).parent.parent

GPU = 'RTX-PRO-6000' #the same card as every other run
CPU_CORES = 8.0
MEMORY_MB = 32768
TIMEOUT_SECONDS = 60 * 60 #three short lora runs and two generation passes take minutes

image = (
    modal.Image.debian_slim(python_version='3.11')
    #blackwell cards need cuda 12.8 builds, the default pypi wheels are older and would fail to launch a kernel
    .pip_install('torch', index_url='https://download.pytorch.org/whl/cu128')
    #trl and peft pinned to the versions the notebook was tested with, their apis change between releases
    .pip_install(
        'papermill==2.7.0', 'ipykernel', 'transformers==4.57.1', 'trl==0.27.2', 'peft==0.17.1', 'accelerate',
        'datasets>=4.7,<5', 'huggingface_hub', 'safetensors', 'pandas', 'matplotlib'
    )
    #copy=True bakes the persona data and the eval prompts into the image, a detached run needs nothing local
    .add_local_dir(REPO_DIR / 'data' / 'persona_lora', remote_path='/root/data/persona_lora', copy=True)
    .add_local_dir(REPO_DIR / 'eval_sets', remote_path='/root/eval_sets', copy=True)
    .add_local_file(NOTEBOOK, remote_path='/root/persona.ipynb', copy=True)
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME, image=image)

#the data and prompts are in the image rather than next to the notebook's working directory. everything else keeps
#the notebook's own values, which already suit a single large card
PARAMETERS = {
    'PERSONA_DATA': '/root/data/persona_lora/ramsay_style_preferences.json',
    'EVAL_SET': '/root/eval_sets/persona_style_eval_40.json',
}

#the same 1.1 as the main comparison's ablation, for the base model and every adapter alike. at 5e-5 the adapters
#write the persona but greedy decoding drags many answers into a repeated line
PENALTY_PARAMETERS = {'REPETITION_PENALTY': 1.1, 'GENERATE_NAME': 'generate_rep1.1'}


@app.function(gpu=GPU, cpu=CPU_CORES, memory=MEMORY_MB, timeout=TIMEOUT_SECONDS, volumes={VOLUME_PATH: volume})
def run(jobs):
    import logging

    import papermill

    #papermill forwards each cell's stdout through python logging at info level, this handler shows it in the logs
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    volume.reload()
    Path(REMOTE_TASK_DIR).mkdir(parents=True, exist_ok=True)

    #(parameters, run name) pairs one after another in this container. a detached launch only keeps the last remote
    #call alive once the local process is gone, so both decodings go into a single call
    for parameters, run_name in jobs:
        executed = f'{REMOTE_TASK_DIR}/{run_name}_executed.ipynb'
        print(f'running {run_name} with parameters {parameters}')
        try:
            papermill.execute_notebook(
                '/root/persona.ipynb', executed,
                parameters=parameters,
                cwd=REMOTE_TASK_DIR, #relative paths inside the notebook resolve here, on the volume
                kernel_name='python3',
                progress_bar=False,
                log_output=True, #every cell's stdout goes to the modal logs, so a detached run is still followable
            )
        finally:
            #committed even when a cell raises, so the partially executed notebook and whatever results exist are kept
            volume.commit()
            print(f'results committed to {REMOTE_TASK_DIR}')


@app.local_entrypoint()
def main(penalty: bool = False):
    jobs = [(PARAMETERS, 'persona')]
    if penalty:
        jobs.append(({**PARAMETERS, **PENALTY_PARAMETERS}, 'persona_rep1.1'))
    run.remote(jobs)
