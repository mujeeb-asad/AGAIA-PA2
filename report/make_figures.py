# report figures -> compact, vector versions of the training and persona plots, sized for one column of the report
#
# everything is read from the results already in the repo, nothing is trained or judged here. run from the repo root:
#     python report/make_figures.py

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
TRAIN_DIR = REPO / 'train/modal_run/outputs/train'
PERSONA_DIR = REPO / 'persona/modal_run/outputs/persona'
SUMMARY_DIR = REPO / 'summary/outputs/summary'
FIGURE_DIR = Path(__file__).resolve().parent / 'figures'
FIGURE_DIR.mkdir(exist_ok=True)

plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7, 'legend.fontsize': 6,
                     'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'lines.linewidth': 1.1, 'pdf.fonttype': 42})

#the three from scratch dpo runs, the colour of each kept the same in every panel
DPO_RUNS = {'dpo_hh': ('DPO-HH', '#1f77b4'), 'dpo_uf': ('DPO-UF', '#d62728'),
            'dpo_uf_lr5e-6': ('DPO-UF lr 5e-6', '#ff7f0e')}

# figure 1 -> sft loss, then the dpo validation loss, reward accuracy and margin of all three runs
fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.75))

sft = pd.read_csv(TRAIN_DIR / 'sft_history.csv')
sft_val = sft.dropna(subset=['val_loss'])
axes[0].plot(sft['step'], sft['train_loss'], color='#7f7f7f', alpha=0.6, linewidth=0.8, label='train (10-step mean)')
axes[0].plot(sft_val['step'], sft_val['val_loss'], marker='o', markersize=2.5, color='black', label='validation')
best = sft_val.loc[sft_val['val_loss'].idxmin()]
axes[0].plot(best['step'], best['val_loss'], marker='*', markersize=9, color='black', linestyle='none')
axes[0].set_title('(a) SFT loss')
axes[0].set_ylabel('nats per response token')
axes[0].legend(frameon=False, loc='lower left')

for run, (label, colour) in DPO_RUNS.items():
    history = pd.read_csv(TRAIN_DIR / f'{run}_history.csv').dropna(subset=['val_loss'])
    best_step = json.loads((TRAIN_DIR / f'{run}_done.json').read_text())['best_step']
    best = history[history['step'] == best_step].iloc[-1]
    for axis, metric in zip(axes[1:], ['val_loss', 'val_reward_accuracy', 'val_reward_margin']):
        #step 0 accuracy is exactly 0 by construction (policy = reference), so the accuracy panel starts at step 50
        rows = history[history['step'] > 0] if metric == 'val_reward_accuracy' else history
        axis.plot(rows['step'], rows[metric], marker='o', markersize=2, color=colour, label=label)
        #the kept checkpoint, best validation reward accuracy
        axis.plot(best['step'], best[metric], marker='*', markersize=9, color=colour, markeredgecolor='black',
                  markeredgewidth=0.4, linestyle='none')
axes[1].set_title('(b) DPO validation loss')
axes[2].set_title('(c) validation reward accuracy')
axes[3].set_title('(d) validation reward margin')
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(0.63, 0.97))
for axis in axes:
    axis.set_xlabel('optimizer step')
    axis.grid(alpha=0.25, linewidth=0.5)
fig.tight_layout(pad=0.3, w_pad=0.8)
fig.savefig(FIGURE_DIR / 'training.pdf', bbox_inches='tight')

# figure 2 -> the persona grid: validation margin over training, and the judged style against quality
selection = pd.read_csv(PERSONA_DIR / 'generate' / 'selection.csv')
persona = pd.read_csv(SUMMARY_DIR / 'persona_vs_base.csv')
rate_colour = {1e-5: '#9ecae1', 2e-5: '#4292c6', 5e-5: '#fd8d3c', 1e-4: '#d62728'}
RATE_NAMES = {1e-5: '1e-5', 2e-5: '2e-5', 5e-5: '5e-5', 1e-4: '1e-4'}

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2), gridspec_kw={'width_ratios': [1, 1.15]})
for run in selection.itertuples():
    history = pd.read_csv(PERSONA_DIR / f'{run.run}_history.csv').dropna(subset=['eval_loss'])
    style = '-' if run.rpo_alpha else '--'
    axes[0].plot(history['step'], history['eval_rewards/margins'], linestyle=style, color=rate_colour[run.learning_rate],
                 label=f'lr {RATE_NAMES[run.learning_rate]}' + (' + RPO' if run.rpo_alpha else ''))
    kept = history[history['step'] == run.best_step].iloc[-1]
    axes[0].plot(kept['step'], kept['eval_rewards/margins'], marker='*', markersize=7,
                 color=rate_colour[run.learning_rate], markeredgecolor='black', markeredgewidth=0.4, linestyle='none')
axes[0].set_title('(a) LoRA-DPO validation reward margin (dashed: plain DPO)')
axes[0].set_xlabel('optimizer step')
axes[0].set_ylabel('reward margin')
axes[0].legend(frameon=False, ncol=2, loc='upper left')
axes[0].grid(alpha=0.25, linewidth=0.5)

for row in persona.itertuples():
    learning_rate = float(row.adapter.split('_lr')[1].split('_')[0])
    marker = 'o' if row.decoding == 'persona' else 's'
    axes[1].scatter(row.style_win_rate, row.quality_win_rate, marker=marker, s=26 if row.selected else 16,
                    facecolor=rate_colour[learning_rate] if 'rpo' in row.adapter else 'white',
                    edgecolor='black' if row.selected else rate_colour[learning_rate],
                    linewidth=1.3 if row.selected else 0.9, zorder=3)
    if row.selected:
        name = row.adapter.replace('persona_', '').replace('_rpo1', ' + RPO').replace('lr0.0001', 'lr 1e-4').replace('lr2e-05', 'lr 2e-5')
        axes[1].annotate(f'{name}\n({"greedy" if row.decoding == "persona" else "rep. 1.1"}, selected)',
                         (row.style_win_rate, row.quality_win_rate), fontsize=6, xytext=(-4, 7),
                         textcoords='offset points', ha='right' if row.style_win_rate > 0.8 else 'left')
axes[1].axhline(0.5, color='grey', linewidth=0.6, linestyle=':')
axes[1].axvline(0.5, color='grey', linewidth=0.6, linestyle=':')
axes[1].set_xlim(0.4, 1.0)
axes[1].set_ylim(0.15, 0.75)
axes[1].set_xlabel('style win rate vs base (ties ½)')
axes[1].set_ylabel('quality win rate vs base (ties ½)')
axes[1].set_title('(b) judged style vs quality, 40 prompts each')
#a legend of the encodings rather than of the 16 points
handles = [plt.Line2D([], [], marker='o', linestyle='none', markerfacecolor='white', markeredgecolor='black', label='greedy'),
           plt.Line2D([], [], marker='s', linestyle='none', markerfacecolor='white', markeredgecolor='black', label='rep. penalty 1.1'),
           plt.Line2D([], [], marker='o', linestyle='none', markerfacecolor='grey', markeredgecolor='grey', label='filled: + RPO')]
axes[1].legend(handles=handles, frameon=False, loc='lower left')
axes[1].grid(alpha=0.25, linewidth=0.5)
fig.tight_layout(pad=0.3, w_pad=1.0)
fig.savefig(FIGURE_DIR / 'persona.pdf', bbox_inches='tight')
print(f'wrote {FIGURE_DIR / "training.pdf"} and {FIGURE_DIR / "persona.pdf"}')
