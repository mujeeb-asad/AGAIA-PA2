# modal runner for the generation notebook -> base, sft and dpo answers on the fixed eval sets, plus the pair files
#
# launch it detached, the same way the training run was. it needs the training run to have finished on this volume
# first, since it loads the checkpoints that run saved:
#     modal run --detach generate/run_generate_modal.py
#
# --scale only raises the generation batch size for a 96gb card. decoding is greedy, so the batch size changes
# speed and nothing else:
#     modal run --detach generate/run_generate_modal.py --scale
#
# --penalty is the decoding ablation: the same models and prompts with a modest repetition penalty, written to its
# own folder so the main greedy results stay untouched. combine it with --scale:
#     modal run --detach generate/run_generate_modal.py --scale --penalty
#
# --uf-retry answers alpaca eval with the retried ultrafeedback run (dpo_uf_lr5e-6) and sft, under both decodings,
# each into its own folder. it needs the retry's training run to have finished:
#     modal run --detach generate/run_generate_modal.py --scale --uf-retry
#
# pull the results back afterwards (executed notebook, generation jsons, pair files):
#     modal volume get agaia /pa2/generate/ ./generate/modal_run
#
# watch a detached run:  modal app logs agaia-pa2-generate

from pathlib import Path

import modal

APP_NAME = 'agaia-pa2-generate'
VOLUME_NAME = 'agaia' #the same volume the training run wrote to
VOLUME_PATH = '/mnt/agaia'
#the notebook runs with this as its working directory, so its relative './outputs' lands on the volume and its
#'../train/outputs/train/checkpoints' resolves to the checkpoints the training run left there
REMOTE_TASK_DIR = f'{VOLUME_PATH}/pa2/generate'
NOTEBOOK = Path(__file__).with_name('generate.ipynb')
REPO_DIR = Path(__file__).parent.parent

GPU = 'RTX-PRO-6000' #the same card as training, so one gpu type covers every reported run
CPU_CORES = 8.0
MEMORY_MB = 32768
TIMEOUT_SECONDS = 60 * 60 #one hour, five passes over 300 prompts take minutes

image = (
    modal.Image.debian_slim(python_version='3.11')
    #blackwell cards need cuda 12.8 builds, the default pypi wheels are older and would fail to launch a kernel
    .pip_install('torch', index_url='https://download.pytorch.org/whl/cu128')
    .pip_install(
        'papermill==2.7.0', 'ipykernel', 'transformers>=4.43.0,<5', 'huggingface_hub', 'accelerate',
        'safetensors', 'sentencepiece', 'pandas'
    )
    .env({'PYTHONPATH': '/root'}) #the kernel starts in the volume directory, this is what lets it import alignment
    #copy=True bakes the code and the eval prompts into the image, so a detached run needs nothing from the laptop
    .add_local_dir(REPO_DIR / 'alignment', remote_path='/root/alignment', copy=True)
    .add_local_dir(REPO_DIR / 'eval_sets', remote_path='/root/eval_sets', copy=True)
    .add_local_file(NOTEBOOK, remote_path='/root/generate.ipynb', copy=True)
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME, image=image)

#the eval prompts are in the image rather than next to the notebook's working directory, every run needs this one
SPEC_PARAMETERS = {'EVAL_SET_DIR': '/root/eval_sets'}

#greedy decoding gives the same text at any batch size, so this is purely a speed change
SCALED_PARAMETERS = {**SPEC_PARAMETERS, 'GEN_BATCH_SIZE': 64}

#greedy decoding loops on roughly a tenth of the answers. 1.1 is the modest end of the usual range, every model in a
#comparison gets the same value, and the separate folder keeps this a labelled ablation next to the main greedy run
PENALTY_PARAMETERS = {'REPETITION_PENALTY': 1.1, 'OUTPUT_DIR': './outputs/generate_rep1.1'}

#the retried ultrafeedback run against sft on alpaca eval. sft is generated again in each folder rather than copied,
#greedy decoding makes it identical to the first run's answers, which is also a check of the setup's determinism
UF_RETRY_PLAN = {'GENERATION_PLAN': [['sft', 'alpaca_eval'], ['dpo_uf_lr5e-6', 'alpaca_eval']],
                 'PAIR_PLAN': [['dpo_uf_lr5e-6', 'alpaca_eval']]}


@app.function(gpu=GPU, cpu=CPU_CORES, memory=MEMORY_MB, timeout=TIMEOUT_SECONDS, volumes={VOLUME_PATH: volume})
def run(jobs):
    import logging

    import papermill

    #papermill forwards each cell's stdout through python logging at info level. without a configured handler only
    #stderr reaches the modal logs and the sanity check answers would be silently dropped
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    volume.reload() #picks up the checkpoints the training run committed
    Path(REMOTE_TASK_DIR).mkdir(parents=True, exist_ok=True)

    #(parameters, run name) pairs run one after another in this one container. a detached launch only keeps the
    #last remote call alive once the local process is gone, so several decodings go into a single call
    for parameters, run_name in jobs:
        #failing here is much clearer than a missing folder error from inside the notebook
        checkpoint_dir = f'{VOLUME_PATH}/pa2/train/outputs/train/checkpoints'
        plan = parameters.get('GENERATION_PLAN', [['sft'], ['dpo_hh'], ['dpo_uf']])
        missing = [f'{row[0]}_best' for row in plan if row[0] != 'qwen2.5-0.5b-base'
                   and not Path(f'{checkpoint_dir}/{row[0]}_best').exists()]
        if missing:
            raise RuntimeError(f'run the training notebook first, these checkpoints are missing from the volume: {missing}')

        executed = f'{REMOTE_TASK_DIR}/{run_name}_executed.ipynb' #one per run, a later run keeps the earlier notebooks
        print(f'running {run_name} with parameters {parameters}')

        try:
            papermill.execute_notebook(
                '/root/generate.ipynb', executed,
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
def main(scale: bool = False, penalty: bool = False, uf_retry: bool = False):
    parameters = SCALED_PARAMETERS if scale else SPEC_PARAMETERS
    if uf_retry:
        run.remote([
            ({**parameters, **UF_RETRY_PLAN, 'OUTPUT_DIR': './outputs/generate_dpo_uf_lr5e-6'}, 'generate_dpo_uf_lr5e-6'),
            ({**parameters, **UF_RETRY_PLAN, 'REPETITION_PENALTY': 1.1,
              'OUTPUT_DIR': './outputs/generate_rep1.1_dpo_uf_lr5e-6'}, 'generate_rep1.1_dpo_uf_lr5e-6'),
        ])
    elif penalty:
        run.remote([({**parameters, **PENALTY_PARAMETERS}, 'generate_rep1.1')])
    else:
        run.remote([(parameters, 'generate')])
