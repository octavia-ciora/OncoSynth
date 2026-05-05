## Reproducing the paper experiments

This section describes how to reproduce the experimental workflow used in our paper. In addition to OncoSynth, the paper experiments include TabDiff and CTGAN as baseline generators as well as evaluation of statistical fidelity and clinical utility.

The paper experiments use the SEER-derived lung and breast cancer cohorts.


### 1. Prepare the cohort
```bash
conda activate gen_env

python generation/prepare.py --cohort <cohort_name> --seed <seed>

# Example: lung cancer cohort
python generation/prepare.py --cohort lung --seed 0

# Example: breast cancer cohort
python generation/prepare.py --cohort breast --seed 0
```

### 2. Run OncoSynth
```bash
python generation/train_oncosynth.py --cohort <cohort_name> --gpu <gpu_id> --cohort_seed <seed>
python generation/generate_oncosynth.py --cohort <cohort_name> --gpu <gpu_id> --cohort_seed <seed>

```

### 3. Run baselines
##### CTGAN:
```bash
python generation/generate_ctgan.py --cohort <cohort_name> --cohort_seed <seed>

```

##### TabDiff:
```bash
python generation/train_tabdiff.py --cohort <cohort_name> --gpu <gpu_id> --cohort_seed <seed>
python generation/generate_tabdiff.py --cohort <cohort_name> --gpu <gpu_id> --cohort_seed <seed>

```

### 4. Evaluation:
Create evaluation environment:
```bash
./create_eval_env.sh
conda activate eval_env
```

Evaluate statistical fidelity:
```bash
python evaluation/evaluate_fidelity.py --cohort_folder <cohort_name>_<seed>

python evaluation/evaluate_fidelity.py --cohort_folder lung_0
```

```bash
python evaluation/evaluate_utility.py --cohort_folder <cohort_name>_<seed> --horizon <horizon>

# Example: evaluate the lung cancer cohort for horizon 120
python evaluation/evaluate_utility.py --cohort_folder lung_0 --horizon 120
```
The ```--horizon``` argument specifies the prediction horizon for utility evaluation. For example, 120 corresponds to 120 months.

### 5. Run multiple seeds:
Scripts for running the generators across multiple seeds are provided in ```generation/```:

```bash
bash generation/run_oncosynth_for_seeds.sh
bash generation/run_ctgan_for_seeds.sh
bash generation/run_tabdiff_for_seeds.sh


# Example: run OncoSynth for the lung cancer cohort over five seeds
bash generation/run_oncosynth_for_seeds.sh lung 0 1 2 3 4
```

For evaluation over multiple seeds, run:
```bash
bash evaluation/run_eval_for_seeds.sh

# Example: evaluate the lung cancer cohort for five seeds and four horizons
bash evaluation/run_eval_for_seeds.sh lung --horizons 36 60 84 120 --seeds 0 1 2 3 4
```
