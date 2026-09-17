# tse-research

Code for two lines of work:

1. Pulling one person's voice out of a recording where several people talk at once, and telling when that person has stopped talking.
2. Retraining a noise-removal model so it cleans up audio without changing how the person's voice sounds.

## Before you run anything

This repo holds only the code written on top of the AD-FlowTSE project. It does not
include AD-FlowTSE itself. Clone that next to this folder and put it on your path, or
the imports starting with `models.`, `utils.`, and `data.` will not resolve:

```bash
git clone https://github.com/aleXiehta/AD-FlowTSE.git
export PYTHONPATH="$PWD/AD-FlowTSE:$PWD/tse-research"
```

Then install the packages:

```bash
pip install -r requirements.txt
```

Run every command from the top of this folder.

### Telling the scripts where your audio lives

Scripts that need a dataset read these. The defaults are relative folders that probably
do not exist on your machine, so set the ones you need:

| Variable | Used for |
| --- | --- |
| `LIBRIMIX_DIR` | The audiobook-based test set. |
| `DIHARD_DIR` | The real-world conversation recordings. |
| `VQ_ATTRIBUTES_CSV` | The spreadsheet of voice descriptions, for the four `vq_*` scripts. |

For example:

```bash
export LIBRIMIX_DIR=/path/to/Libri2Mix
export DIHARD_DIR=/path/to/dihard/data
```

---

## configs/

Settings files. Open them and edit the paths at the top before training.

- **`speaker_gate.yaml`** — Settings for training the is-this-person-talking check.
- **`t_predicter_target_absent.yaml`** — Settings for retraining the how-much-of-this-is-them estimator.
- **`udit_target_absent.yaml`** — Settings for retraining the main voice-separating model.

## core/

- **`inference.py`** — The main tool. Give it a recording and a short clip of one person, and it writes out just that person's voice.
  `python core/inference.py --config CONFIG.yaml --mixture mix.wav --enrollment clip.wav --output out.wav`
  Add `--alpha 0.5` to set the strength by hand, or `--t_predicter_ckpt FILE.ckpt` to let the model decide.
- **`speaker_gate.py`** — The is-this-person-talking check itself. Not run on its own; other scripts import it.

## training/

All three take a settings file and run until finished.

- **`train_speaker_gate.py`** — Teaches the yes/no check whether the person is talking.
  `python training/train_speaker_gate.py --config configs/speaker_gate.yaml`
- **`train_t_predicter_target_absent.py`** — Retrains the how-much-of-this-is-them estimator, adding examples where the person is not there at all.
  `python training/train_t_predicter_target_absent.py --config configs/t_predicter_target_absent.yaml`
- **`train_udit_target_absent.py`** — Same idea, but retrains the big voice-separating model. Needs several graphics cards.
  `python training/train_udit_target_absent.py --config configs/udit_target_absent.yaml`

## evaluation/

None of these take arguments. Edit the file paths near the top, then run.

- **`eval_speaker_gate_librimix.py`** — Scores the yes/no check on the audiobook test set.
  `python evaluation/eval_speaker_gate_librimix.py`
- **`eval_speaker_gate_dihard.py`** — Scores the same check on real conversations.
  `python evaluation/eval_speaker_gate_dihard.py`
- **`eval_t_predicter_librimix.py`** — Compares the original and retrained estimator on the audiobook set.
  `python evaluation/eval_t_predicter_librimix.py`
- **`eval_t_predicter_dihard.py`** — The same comparison on real conversations. Run `experiments/extract_dihard_alpha.py` first.
  `python evaluation/eval_t_predicter_dihard.py`
- **`similarity_postprocess.py`** — Takes finished output and mutes the stretches that do not sound like the person. Run `experiments/extract_dihard_alpha.py` first.
  `python evaluation/similarity_postprocess.py`

## experiments/

- **`enrollment_length.py`** — Asks how long the sample clip of the person needs to be. Tries 0.5 up to 7 seconds.
  `python experiments/enrollment_length.py --librimix_dir $LIBRIMIX_DIR --n_samples 1000`
  Add `--noise_ratio 0.4` to make the sample clip noisy.
- **`noisy_enrollment.py`** — Compares a clean sample clip against ones with 20% and 40% noise added.
  `python experiments/noisy_enrollment.py --librimix_dir $LIBRIMIX_DIR --n_samples 3000`
- **`wrong_enrollment.py`** — Feeds it the wrong person's clip on purpose, to see what breaks.
  `python experiments/wrong_enrollment.py --librimix_dir $LIBRIMIX_DIR --n_samples 50`
- **`multi_enrollment.py`** — Checks whether two sample clips of the person beat one.
  `python experiments/multi_enrollment.py --n_samples 250`
- **`multi_enrollment_dihard.py`** — The same two-clip test on real conversations.
  `python experiments/multi_enrollment_dihard.py`
- **`extract_dihard.py`** — Runs the main tool over real conversation recordings, one person at a time.
  `python experiments/extract_dihard.py`
- **`extract_dihard_gated.py`** — The same, but also mutes the parts where the person is not talking. Saves both versions side by side.
  `python experiments/extract_dihard_gated.py`
- **`extract_dihard_alpha.py`** — Picks a clean three-second sample of each person from a conversation recording, then separates that person out twice: once with a fixed setting and once letting the model choose the setting. Writes to `dihard_results/<recording>/<speaker>/`, which is what the two DIHARD evaluation scripts and `analysis/report_dihard.py` read. Add `--sweep` to repeat the whole thing across five solver settings, saved separately under `sweeps/`.
  `python experiments/extract_dihard_alpha.py --file-id DH_EVAL_0012`
- **`tse_vs_reuse.py`** — Sets up a comparison against a general noise-removal model. Prepares the audio; the other half runs in a notebook.
  `python experiments/tse_vs_reuse.py --n_samples 100`

## datagen/

Builds the audio you need before anything else will run.

- **`generate_librimix_test.py`** — Downloads and builds the audiobook test set. Large download.
  `python datagen/generate_librimix_test.py --output_dir /path/to/output`
  Add `--keep_cache` to pick up where a stopped run left off.
- **`generate_librimix_train.py`** — The same, for the training audio.
  `python datagen/generate_librimix_train.py --output_dir /path/to/output`
- **`regenerate_librimix.sh`** — Does the test-set build as a shell script instead.
  `BUILD_DIR=/tmp/build DEST_DIR=/path/to/output bash datagen/regenerate_librimix.sh`
- **`noise_pipeline.py`** — Adds a measured amount of noise to clean audio.
  `python datagen/noise_pipeline.py --input_dir clean/ --output_dir noisy/`
- **`prepare_reuse_experiment.py`** — Gathers audio into the layout the noise-removal comparison expects.
  `python datagen/prepare_reuse_experiment.py`
- **`prepare_reuse_libri1mix.py`** — The same, for the one-person-plus-noise version.
  `python datagen/prepare_reuse_libri1mix.py --n_samples 50`

## vq_finetune/

Retraining a noise-removal model so it does not flatten how a voice sounds. Run these
from inside this folder.

- **`config.yaml`** — All the settings. Pick which version to train with `experiment.arm`.
- **`train.py`** — Does the retraining. Five versions to choose from: two that ignore how the voice sounds, for comparison, and three that take it into account.
  `python train.py --config config.yaml --arm B1`
- **`evaluate.py`** — Compares a retrained model against the untouched one on the same audio.
  `python evaluate.py --ckpt model.ckpt`
- **`evaluate_independent.py`** — Scores it on measures it was never trained on, so the result is not circular.
  `python evaluate_independent.py --ckpt model.ckpt`
- **`evaluate_ood.py`** — Scores it on audio unlike anything it trained on, where no clean copy exists to compare against.
  `python evaluate_ood.py --ood-dir /path/to/audio --ckpt model.ckpt`
- **`make_subset.py`** — Cuts a big training list down to a smaller one that still has the same mix of hard and easy cases.
  `python make_subset.py --src meta.tsv --dst small.tsv --n 10000`
- **`test_local.py`** — A quick check that everything works, on a laptop, with no graphics card.
  `python test_local.py`
- **`dataloader.py`**, **`discriminator.py`**, **`semamba_discriminator.py`**, **`semamba_loss.py`**, **`vq_metric.py`** — Parts used by `train.py`. Not run on their own.

## voice_quality/

Describing how a voice sounds, and whether cleaning it up changed that.

- **`evaluate_voice_quality.py`** — Describes the same voice before and after cleanup, side by side.
  `python voice_quality/evaluate_voice_quality.py --n_samples 10`
- **`evaluate_dihard_voice_quality.py`** — The same, for long real-world recordings. Saves the worst cases as separate files.
  `python voice_quality/evaluate_dihard_voice_quality.py`
- **`evaluate_reuse_experiment.py`** — Collects the noise-removal results into one spreadsheet.
  `python voice_quality/evaluate_reuse_experiment.py`
- **`voice_quality_correction.py`** — Blends the noisy and cleaned audio to land on whichever mix sounds most like the original person.
  `python voice_quality/voice_quality_correction.py --enrollment clip.wav --noisy in.wav --enhanced clean.wav --output out.wav`
- **`compare_tse_reuse.py`** — Compares the one-person tool against the general noise remover.
  `python voice_quality/compare_tse_reuse.py --results_dir tse_vs_reuse_results`
- **`vq_significance.py`** — Checks whether the differences in the voice descriptions are real or chance.
  `python voice_quality/vq_significance.py`
- **`vq_preservation.py`** — Across all 25 voice descriptions, asks which version stayed closer to the original voice.
  `python voice_quality/vq_preservation.py`
- **`vq_revised.py`** — The same check, narrowed to the descriptions that published work links to depression.
  `python voice_quality/vq_revised.py`
- **`vq_label_check.py`** — Prints both ways of scoring side by side so you can check the numbers by hand.
  `python voice_quality/vq_label_check.py`

## analysis/

Follow-ups on a finished run. Neither takes arguments; both read result folders left behind by earlier scripts.

- **`report_dihard.py`** — Reads the separated audio for one DIHARD recording and writes a text report: how long each speaker talks, loudness and other simple measurements for the mixture and each separated file, and how much the two alpha settings differ. The recording name is set at the top of the file (`FILE_ID`); run `experiments/extract_dihard_alpha.py` on that recording first, then this saves `report.md` next to its output.
  `python analysis/report_dihard.py`
- **`save_worst_samples.py`** — Picks the five worst-scoring samples from the noisy-enrollment run, re-runs the model on each one with both a clean and a noisy enrollment, and saves the mixture, the true speech, both enrollments, and both separated files into one folder per sample, plus a text file with the scores. Needs a GPU and a model checkpoint. Writes to `noisy_enrollment_results/worst_5_inspection/`.
  `python analysis/save_worst_samples.py`

## notebooks/

For Google Colab, where the graphics cards are. Upload one and run the cells in order.
Each clones what it needs at the start.

- **`speaker_gate_training.ipynb`** — Trains the is-this-person-talking check.
- **`t_predicter_target_absent.ipynb`** — Retrains the how-much-of-this-is-them estimator.
- **`vq_finetune_datagen.ipynb`** — Builds the training audio for the retraining work.
- **`vq_finetune_train.ipynb`** — Runs the full retraining.
- **`vq_finetune_smoke_test.ipynb`** — A short run to confirm it works before committing to a long one.
- **`reuse_experiment.ipynb`** — Runs the noise-removal model over the prepared audio.
- **`reuse_libri1mix.ipynb`** — The same, for one person plus background noise.
- **`reuse_3000.ipynb`** — The same, at a larger scale.
- **`voice_quality_3000.ipynb`** — Describes how the voices sound across that larger run.

Three of these are plain Python files rather than notebooks. Paste them into Colab a
section at a time, or run them there directly.

- **`enrollment_length_colab.py`** — The sample-clip-length experiment.
- **`reuse_inference_colab.py`** — Runs the noise-removal model over a folder.
  `python reuse_inference_colab.py --input_dir noisy/ --output_dir cleaned/`
- **`voice_quality_reuse_colab.py`** — A short noise-removal run for the voice-description comparison.
