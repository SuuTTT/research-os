# Structural-Entropy-Enhanced N-BEATS, N-BEATSx, and TimesNet for Long-Horizon Forecasting

**Research idea.** Integrate structural entropy into N-BEATS, N-BEATSx, and TimesNet to improve long-horizon forecasting.

**Goal.** Identify the current state of the art, benchmark protocols, official code, datasets, metrics, reproduction pitfalls, and a fair first experimental plan.

---

## 1. Problem statement

A sharper formulation of the idea is:

> Use structural entropy as a structural prior, regularizer, or auxiliary objective over variable relationships, scale hierarchies, and exogenous-information organization, and inject it into N-BEATS, N-BEATSx, and TimesNet to improve long-horizon forecasting stability, generalization, and robustness across prediction horizons.

The motivation is straightforward. N-BEATS uses basis expansion, deep fully connected blocks, and doubly residual stacking for interpretable time-series forecasting [1,2]. N-BEATSx extends N-BEATS to explicitly incorporate exogenous variables and to decompose how external information interacts with trend and seasonal components [3,4]. TimesNet maps one-dimensional temporal variation into two-dimensional multi-period representations and models intraperiod and interperiod variation through TimesBlock [5,6]. All three already benefit from some form of structured decomposition, but none of their official papers or official code explicitly optimizes **structural complexity itself** as a first-class objective.

In the current long-term time-series forecasting (LTSF) context, the safest research claim is not “universal improvement on all time-series tasks.” A more defensible claim is:

> Under a strictly comparable benchmark protocol, test whether structural entropy improves: (i) organization of variable-level representations, (ii) stability of multiscale or multiperiod decomposition, and (iii) use of exogenous variables without simply injecting more noise.

This framing is aligned with current concerns in the field. The Time Series Library (TSLib) maintainers explicitly split the long-term forecasting leaderboard into **Look-Back-96** and **Look-Back-Searching** categories because different papers used inconsistent historical look-back lengths [19]. TSLib also warns that some older benchmarks may no longer be meaningful for measuring current research progress and points to the “Accuracy Law” discussion of saturated forecasting tasks [19,23]. Therefore, a structural-entropy paper should emphasize **mechanistic gain, protocol fairness, and variance/stability analysis**, not just headline score deltas.

The research boundary should be split into two tracks:

1. **Traditional supervised multivariate long-term forecasting**, using the legacy ETT, Electricity/ECL, Traffic, Weather, Exchange, and ILI benchmarks.
2. **Forecasting with exogenous variables**, where N-BEATSx and TimeXer are the most relevant comparison points [3,14,15].

These tracks should not be merged into a single “overall SOTA” table because their task definitions differ.

---

## 2. SOTA and strong related methods

The table below separates models directly related to the three target backbones from strong baselines that should still be included in a fair comparison. “Training budget” means either the official paper/repo configuration or a reproducible starting point; it should not be interpreted as a strict FLOPs-equivalent budget.

| Paper / model | Year | Venue | Method | Representative benchmark | Main metric | Representative result / claim | Training-budget note | Official code | License status |
|---|---:|---|---|---|---|---|---|---|---|
| N-BEATS | 2020 | ICLR | Deep MLP with doubly residual stacking and basis expansion | M3, M4, Tourism | OWA, sMAPE, MASE | ICLR abstract reports state-of-the-art results and about 3% improvement over the M4 competition winner [1]. | Paper/repo designed to reproduce paper experiments; original implementation is PyTorch [2]. | [ServiceNow/N-BEATS](https://github.com/ServiceNow/N-BEATS) | CC BY-NC 4.0 / restrictive non-commercial code license in source headers [2]. |
| N-BEATSx | 2023 | International Journal of Forecasting | N-BEATS with exogenous-variable encoders | Electricity price forecasting (EPF), multiple markets | MAE, RMSE, sMAPE, rMAE | Abstract reports nearly 20% improvement over original N-BEATS and up to 5% over specialized statistical/ML methods [3]. | Official reproduction command uses `--hyperopt_iters 1500 --n_val_weeks 52 --random_validation 0` [4]. | [cchallu/nbeatsx](https://github.com/cchallu/nbeatsx) | MIT [4]. |
| TimesNet | 2023 | ICLR | Adaptive multiperiod discovery, 1D-to-2D temporal-variation modeling, TimesBlock | Long/short forecasting, imputation, classification, anomaly detection | MSE, MAE, OWA depending on task | OpenReview abstract reports consistent SOTA across five mainstream time-series tasks [5]. | Long-term forecasting scripts available in TimesNet/TSLib; common legacy setup uses `seq_len=96` and horizons 96/192/336/720 [5,6,19]. | [thuml/TimesNet](https://github.com/thuml/TimesNet) and [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library) | MIT for TimesNet repo page / implementation lineage in TSLib [6,19]. |
| PatchTST | 2023 | ICLR | Channel-independent Transformer over time patches | Legacy LTSF | MSE, MAE | Official README reports PatchTST/64 reduces MSE by 21.0% and MAE by 16.7% relative to the best Transformer-based baselines in its comparison [8]. | Official repo provides supervised and self-supervised folders and scripts; data links follow the Autoformer-style benchmark setup [8]. | [yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST) | Apache-2.0 [8]. |
| iTransformer | 2024 | ICLR Spotlight | Inverted Transformer: variates become tokens; attention models multivariate correlations | Multivariate long-term forecasting | MSE, MAE | Official repo states the method was accepted as ICLR 2024 Spotlight and included in TSLib; TSLib lists it in the Look-Back-96 leaderboard [10,11,19]. | Official scripts are provided; default script settings should be logged per dataset rather than assumed globally [11]. | [thuml/iTransformer](https://github.com/thuml/iTransformer) | MIT [11]. |
| TimeMixer | 2024 | ICLR | Pure MLP multiscale decomposition and mixing | Long-term and short-term forecasting | MSE, MAE | Official repo claims consistent SOTA in long- and short-term forecasting with favorable runtime; TSLib lists TimeMixer first in Look-Back-Searching as of its leaderboard snapshot [12,13,19]. | Paper/repo configurations vary by dataset; record epoch, batch size, look-back, and downsampling settings explicitly [13]. | [kwuking/TimeMixer](https://github.com/kwuking/TimeMixer) | Apache-2.0 [13]. |
| TimeXer | 2024 | NeurIPS | Exogenous-aware Transformer with endogenous/exogenous reconciliation | Forecasting with exogenous variables; also multivariate forecasting | MSE, MAE and task-specific metrics | NeurIPS abstract defines forecasting with exogenous variables; official repo reports consistent SOTA on 12 real-world forecasting benchmarks [14,15]. | Repo includes EPF data and scripts; public README does not provide one universal training budget [15]. | [thuml/TimeXer](https://github.com/thuml/TimeXer) | No explicit OSS license was visible on the repository page checked; verify before reuse [15]. |
| N-HiTS | 2023 | AAAI | N-BEATS-family hierarchical interpolation and multi-rate sampling | Long-horizon forecasting | MSE, MAE and relative improvements | Official repo reports 25% average accuracy improvement over recent Transformer architectures and orders-of-magnitude compute-time reduction [16,17]. | Original repo is useful for paper-faithful reproduction; NeuralForecast is often easier for modern experimentation [17,21]. | [cchallu/n-hits](https://github.com/cchallu/n-hits) | No explicit license was visible on the repository page checked; verify before reuse [17]. |
| DLinear / LTSF-Linear | 2023 | AAAI | Simple one-layer linear family; DLinear includes decomposition | Legacy LTSF | MSE, MAE | The paper questions whether Transformers are effective for LTSF and reports simple linear models outperforming more complex Transformer baselines on nine real datasets [18]. | Essential sanity baseline; include it even if the main focus is N-BEATS/TimesNet. | [cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear) | Apache-2.0 [18]. |

**Practical conclusion.** For modifiability, the N-BEATS / N-BEATSx / N-HiTS MLP family is the easiest place to inject hierarchy-aware or variable-structure regularization. For legacy LTSF credibility, TimesNet, PatchTST, iTransformer, TimeMixer, and DLinear should be in the comparison set. For exogenous-variable claims, N-BEATSx and TimeXer are unavoidable.

---

## 3. Official benchmark protocol

For legacy long-term forecasting, the common protocol inherited from Autoformer/TimesNet/TSLib uses:

- Input/look-back length: `seq_len = 96` for most long-term forecasting datasets.
- Prediction horizons: `pred_len ∈ {96, 192, 336, 720}`.
- ILI exception: shorter look-back and horizons are commonly used, e.g. look-back 36 and horizons `{24, 36, 48, 60}` in legacy setups.
- Metrics: MSE as the primary metric and MAE as the secondary metric.

TSLib exposes long-term forecasting through a unified `run.py` interface; its quick-test command explicitly uses `--seq_len 96 --pred_len 96` for long-term forecasting [19]. TSLib’s README also says the long-term forecasting leaderboard was split into Look-Back-96 and Look-Back-Searching because papers used inconsistent look-back lengths [19].

Data splits must be chronological, not random. Standard practice in these loaders is to fit normalization parameters only on the training split and then apply them to validation/test. The Nixtla `datasetsforecast` LongHorizon wrapper explicitly states that its long-horizon datasets are normalized with train-set mean and standard deviation and are partitioned into train, validation, and test splits [20].

For **forecasting with exogenous variables**, do not mix the task with endogenous-only multivariate forecasting. TimeXer defines a practical forecasting paradigm where exogenous variables help predict endogenous variables [14,15]. N-BEATSx is likewise built around using external regressors for electricity price forecasting [3,4]. A structural-entropy improvement to N-BEATSx should therefore be reported as an exogenous-forecasting result, not as a direct headline comparison with endogenous-only LTSF models.

---

## 4. Dataset download and setup instructions

### 4.1 Legacy LTSF through TSLib

The fastest route is to use TSLib’s preprocessed datasets and unified runner.

```bash
git clone https://github.com/thuml/Time-Series-Library.git
cd Time-Series-Library
conda create -n tslib python=3.11
conda activate tslib
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

TSLib’s README says the preprocessed datasets can be obtained from its Google Drive, Baidu Drive, or Hugging Face links and placed under `./dataset` [19]. The same README provides a quick long-term forecasting smoke test:

```bash
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id test_long \
  --model DLinear \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --train_epochs 1 \
  --num_workers 2
```

Recommended initial datasets:

- `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`
- Electricity / ECL
- Traffic
- Weather
- Exchange
- ILI, with its separate shorter-horizon convention

### 4.2 Dataset cross-check through Nixtla `datasetsforecast`

For a cleaner data wrapper, Nixtla’s `datasetsforecast` includes a `LongHorizon` class covering ETT, ECL, Exchange, Traffic, ILI, and Weather. Its source states that each set is normalized using train-data mean and standard deviation and partitioned into train/validation/test splits [20]. This is useful for cross-checking your own loader against TSLib.

### 4.3 N-BEATSx / EPF setup

For N-BEATSx, follow the official repository and its EPF reproduction command. The repo gives the following command for reproducing N-BEATSx forecasts [4]:

```bash
python src/hyperopt_nbeatsx.py \
  --dataset 'NP' \
  --space "nbeats_x" \
  --data_augmentation 0 \
  --random_validation 0 \
  --n_val_weeks 52 \
  --hyperopt_iters 1500 \
  --experiment_id "nbeatsx_0_0"
```

This matters because a single hand-tuned run should not be claimed to beat a fully tuned N-BEATSx baseline unless the tuning budget is comparable.

### 4.4 TimeXer exogenous-forecasting setup

The TimeXer repository includes the short-term electricity price forecasting dataset under `./dataset/EPF` and provides experiment scripts under `./scripts/forecast_exogenous/EPF/` [15]. It also provides links for multivariate and large-scale meteorology datasets [15]. Use TimeXer only in the exogenous-forecasting chapter unless you explicitly separate its multivariate forecasting setup.

---

## 5. Strong baseline repos and commit recommendations

### Unified benchmark harness

Use **TSLib** as the primary harness for legacy LTSF. It includes many canonical baselines, supports five major time-series tasks, and gives a single data/metric/training interface [19]. This is the most convenient way to compare TimesNet, DLinear, iTransformer, TimeMixer, and other models under one framework.

### Paper-faithful official repos

For exact paper reproduction or claims about “official settings,” keep the official repositories separate:

- N-BEATS: [ServiceNow/N-BEATS](https://github.com/ServiceNow/N-BEATS) [2]
- N-BEATSx: [cchallu/nbeatsx](https://github.com/cchallu/nbeatsx) [4]
- TimesNet: [thuml/TimesNet](https://github.com/thuml/TimesNet) [6]
- PatchTST: [yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST) [8]
- iTransformer: [thuml/iTransformer](https://github.com/thuml/iTransformer) [11]
- TimeMixer: [kwuking/TimeMixer](https://github.com/kwuking/TimeMixer) [13]
- TimeXer: [thuml/TimeXer](https://github.com/thuml/TimeXer) [15]
- N-HiTS: [cchallu/n-hits](https://github.com/cchallu/n-hits) [17]
- DLinear / LTSF-Linear: [cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear) [18]

### Maintainable implementation path

Use **Nixtla NeuralForecast** for fast ablation and repeated experiments. Its docs/repo list implementations such as NBEATS, NBEATSx, NHITS, DLinear, PatchTST, iTransformer, TimesNet, and TimeLLM [21]. It is not always paper-identical, but it is useful for rapid structural-entropy probes.

### Commit recommendation

Do not cite a floating `main` branch in a paper. For every repo, immediately after cloning, run:

```bash
git rev-parse HEAD
```

Record the SHA in:

- experiment YAML / JSON config,
- result folder name,
- paper appendix table,
- exact conda/pip lockfile.

This report does not invent commit hashes because repo heads can change and not every source page exposes a stable SHA.

---

## 6. Known reproduction issues

1. **Look-back inconsistency.** TSLib explicitly split long-term forecasting results into Look-Back-96 and Look-Back-Searching because papers used different input lengths [19]. A model tuned at `seq_len=336` should not be compared as if it were a `seq_len=96` result.

2. **Benchmark saturation and tiny gains.** TSLib warns that some older benchmarks may no longer be meaningful for measuring current progress [19]. The Accuracy Law paper argues that forecasting tasks have nonzero lower bounds and that saturated tasks can make small improvements misleading [23].

3. **Data leakage through preprocessing.** Train-only scaling is essential. Using all data to fit scalers, or using random splits, can leak future information. The Nixtla LongHorizon wrapper’s train-normalization note is a good reference point [20].

4. **Budget mismatch.** N-BEATSx’s official reproduction uses 1,500 hyperparameter iterations [4], while many TSLib scripts use shorter fixed-epoch runs. PatchTST, TimeMixer, and iTransformer also use repo-specific scripts. Always state whether the comparison is fixed-budget, official-budget, or search-budget.

5. **Task-definition mixing.** N-BEATS started as a univariate competition-style forecasting model [1,2]. N-BEATSx and TimeXer focus on exogenous variables [3,14]. TimesNet, iTransformer, PatchTST, TimeMixer, and DLinear are often reported on legacy multivariate LTSF [5,8,10,12,18]. Merging these into one undifferentiated “SOTA” table will weaken the claim.

6. **License and reuse risk.** N-BEATS uses a restrictive non-commercial license in the source header [2]. TimeXer and N-HiTS did not show an explicit OSS license on the repository pages checked [15,17]. Verify license terms before building derivative code.

---

## 7. Fairness checklist for claiming improvement

A credible “structural entropy improves forecasting” claim should satisfy the following:

1. **Separate task definitions.** Report endogenous-only multivariate LTSF separately from exogenous-variable forecasting [14,15].
2. **Report Look-Back-96 first.** If look-back search is used, put it in a separate table and label it as such [19].
3. **Use chronological splits and train-only normalization.** Document preprocessing and scaler fitting [20].
4. **Disclose training budget.** Include epochs, batch size, learning rate, patience, search rounds, seeds, early stopping, and GPU type.
5. **Do not compare only against Transformers.** Include DLinear and at least one N-BEATS-family baseline [17,18].
6. **Report mean and variance.** Use multiple seeds and report mean/std, not only the best run.
7. **Disentangle pretraining or external data.** If using pretrained models or additional data, do not mix those results with vanilla supervised baselines.
8. **Add mechanism checks.** Include structural-entropy curves, variable-graph diagnostics, horizon-wise error, and seed stability.
9. **Include ablations.** Compare no-entropy, entropy-as-regularizer, entropy-as-auxiliary-loss, and entropy-guided architecture variants.
10. **Use statistical testing where deltas are tiny.** This is particularly important on saturated datasets [23].

---

## 8. Minimal local benchmark build plan

### Panel A: legacy multivariate LTSF

Use TSLib and run:

- datasets: `ETTh1`, `ETTh2`, ECL, Traffic, Weather, Exchange;
- fixed look-back: `seq_len=96`;
- horizons: `pred_len ∈ {96, 192, 336, 720}`;
- baselines: DLinear, PatchTST, TimesNet, iTransformer, TimeMixer;
- proposed models: `SE-NBEATS` and `SE-TimesNet`.

This panel answers whether structural entropy improves endogenous multivariate long-horizon forecasting under the standard legacy protocol.

### Panel B: exogenous-variable forecasting

Use the N-BEATSx EPF protocol first:

- baseline: N-BEATSx official implementation;
- proposed model: `SE-NBEATSx`;
- optional strong comparison: TimeXer on its EPF exogenous-forecasting setup.

The goal is not to claim a generic LTSF SOTA result, but to test whether structural entropy helps organize exogenous information.

### Logging convention

Every run folder should include:

```text
model=<model>/dataset=<dataset>/seq=<seq_len>/pred=<pred_len>/seed=<seed>/budget=<budget_name>/sha=<git_sha>/
```

Also store:

- `config.yaml`,
- commit SHA of each repo,
- conda/pip lockfile,
- raw prediction arrays,
- metrics by horizon,
- entropy diagnostics.

---

## 9. Suggested first five probes

1. **Insertion-location probe.** Add the same structural-entropy term at different locations: input variable graph, latent block/stack representations, and final forecast aggregation. N-BEATS, N-BEATSx, and TimesNet expose different structural entry points [1,3,5].

2. **Scale-sensitivity probe.** Test whether entropy helps more on high-forecastability datasets such as Weather/Electricity or on harder/noisier datasets such as ETT/Exchange. This distinguishes useful structural bias from simple shrinkage.

3. **Exogenous-variable selection probe.** In N-BEATSx, drop groups of exogenous variables and test whether structural entropy makes the model less sensitive to irrelevant or noisy regressors [3,4].

4. **Fixed protocol vs searched protocol probe.** Report both Look-Back-96 and best-look-back settings. If the entropy method only improves under a searched protocol, the gain may be budget-driven rather than method-driven [19].

5. **Seed and variance probe.** Run multiple seeds and compare mean, standard deviation, and best seed. If structural entropy reduces variance but only slightly improves mean, that is still a meaningful stability contribution.

---

## 10. Criticisms and risks in this area

### 10.1 Legacy LTSF benchmarks may be overused or saturated

TSLib’s README warns that many older benchmarks may no longer be meaningful for evaluating current progress [19]. The Accuracy Law work also argues that time-series forecasting has inherent nonzero error lower bounds and that saturated tasks can make small improvements difficult to interpret [23]. A paper that only reports a 0.00x gain on one or two legacy datasets will be vulnerable unless it includes mechanism and robustness evidence.

### 10.2 Transformer advantage is still contested

The DLinear paper directly questions the effectiveness of Transformer-based LTSF models and shows that simple linear models can outperform complex Transformer baselines in many settings [18]. Therefore, structural-entropy variants must beat simple baselines, not only older Transformers.

### 10.3 Task definitions are often mixed

N-BEATS, legacy multivariate LTSF, and exogenous-variable forecasting are related but not identical. N-BEATSx and TimeXer are especially easy to miscompare because their core contribution is exogenous information handling [3,14,15].

### 10.4 Hidden leakage and budget choices can dominate results

Scaling on all data, random validation splits, undisclosed hyperparameter searches, and seed picking can all produce apparent improvements. N-BEATSx’s official 1,500-iteration reproduction command is a reminder that budget disclosure is not optional [4].

### 10.5 Benchmark coverage criticism

The TFB benchmark paper argues that existing forecasting evaluations suffer from insufficient domain coverage, bias against traditional methods, and inconsistent/inflexible pipelines; it proposes a broader benchmark across 10 domains with statistical, ML, and deep-learning methods [22]. This critique is relevant if the structural-entropy paper relies only on legacy ETT-style benchmarks.

---

## 11. BibTeX entries

```bibtex
@inproceedings{oreshkin2020nbeats,
  title     = {N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting},
  author    = {Oreshkin, Boris N. and Carpov, Dmitri and Chapados, Nicolas and Bengio, Yoshua},
  booktitle = {International Conference on Learning Representations},
  year      = {2020}
}

@article{olivares2023nbeatsx,
  title   = {Neural basis expansion analysis with exogenous variables: Forecasting electricity prices with NBEATSx},
  author  = {Olivares, Kin G. and Challu, Cristian and Marcjasz, Grzegorz and Weron, Rafa{\l} and Dubrawski, Artur},
  journal = {International Journal of Forecasting},
  year    = {2023}
}

@inproceedings{wu2023timesnet,
  title     = {TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis},
  author    = {Wu, Haixu and Hu, Tengge and Liu, Yong and Zhou, Hang and Wang, Jianmin and Long, Mingsheng},
  booktitle = {International Conference on Learning Representations},
  year      = {2023}
}

@inproceedings{nie2023patchtst,
  title     = {A Time Series is Worth 64 Words: Long-term Forecasting with Transformers},
  author    = {Nie, Yuqi and Nguyen, Nam H. and Sinthong, Phanwadee and Kalagnanam, Jayant},
  booktitle = {International Conference on Learning Representations},
  year      = {2023}
}

@inproceedings{liu2024itransformer,
  title     = {iTransformer: Inverted Transformers Are Effective for Time Series Forecasting},
  author    = {Liu, Yong and Hu, Tengge and Zhang, Haoran and Wu, Haixu and Wang, Shiyu and Ma, Lintao and Long, Mingsheng},
  booktitle = {International Conference on Learning Representations},
  year      = {2024}
}

@inproceedings{wang2024timemixer,
  title     = {TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting},
  author    = {Wang, Shiyu and Wu, Haixu and Shi, Xiaoming and Hu, Tengge and Luo, Huakun and Ma, Lintao and Zhang, James Y. and Zhou, Jun},
  booktitle = {International Conference on Learning Representations},
  year      = {2024}
}

@inproceedings{wang2024timexer,
  title     = {TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables},
  author    = {Wang, Yuxuan and Wu, Haixu and Dong, Jiaxiang and Liu, Yong and Qiu, Yunzhong and Zhang, Haoran and Wang, Jianmin and Long, Mingsheng},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2024}
}

@inproceedings{challu2023nhits,
  title     = {Neural Hierarchical Interpolation for Time Series Forecasting},
  author    = {Challu, Cristian and Olivares, Kin G. and Oreshkin, Boris N. and Garza, Federico and Mergenthaler-Canseco, Max and Dubrawski, Artur},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2023}
}

@inproceedings{zeng2023dlinear,
  title     = {Are Transformers Effective for Time Series Forecasting?},
  author    = {Zeng, Ailing and Chen, Muxi and Zhang, Lei and Xu, Qiang},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2023}
}

@article{qiu2024tfb,
  title   = {TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods},
  author  = {Qiu, Xiangfei and Hu, Jilin and Zhou, Lekui and Wu, Xingjian and Du, Junyang and Zhang, Buang and Guo, Chenjuan and Zhou, Aoying and Jensen, Christian S. and Sheng, Zhenli and Yang, Bin},
  journal = {Proceedings of the VLDB Endowment},
  year    = {2024}
}

@article{wang2025accuracylaw,
  title   = {Accuracy Law for the Future of Deep Time Series Forecasting},
  author  = {Wang, Yuxuan and Wu, Haixu and Ma, Yuezhou and Fang, Yuchen and Zhang, Ziyi and Liu, Yong and Wang, Shiyu and Ye, Zhou and Xiang, Yang and Wang, Jianmin and Long, Mingsheng},
  journal = {arXiv preprint arXiv:2510.02729},
  year    = {2025}
}
```

---

## 12. Reference links

[1] N-BEATS OpenReview / ICLR 2020: https://openreview.net/forum?id=r1ecqn4YwB  
[2] N-BEATS official repo: https://github.com/ServiceNow/N-BEATS  
[3] N-BEATSx arXiv: https://arxiv.org/abs/2104.05522  
[4] N-BEATSx official repo: https://github.com/cchallu/nbeatsx  
[5] TimesNet OpenReview / ICLR 2023: https://openreview.net/forum?id=ju_Uqw384Oq  
[6] TimesNet official repo: https://github.com/thuml/TimesNet  
[7] TimesNet / THUML paper PDF: https://ise.thss.tsinghua.edu.cn/~mlong/doc/TimesNet-iclr23.pdf  
[8] PatchTST official repo: https://github.com/yuqinie98/PatchTST  
[9] PatchTST OpenReview / ICLR 2023: https://openreview.net/forum?id=Jbdc0vTOcol  
[10] iTransformer OpenReview / ICLR 2024: https://openreview.net/forum?id=JePfAI8fah  
[11] iTransformer official repo: https://github.com/thuml/iTransformer  
[12] TimeMixer OpenReview / ICLR 2024: https://openreview.net/forum?id=7oLshfEIC2  
[13] TimeMixer official repo: https://github.com/kwuking/TimeMixer  
[14] TimeXer NeurIPS abstract: https://proceedings.neurips.cc/paper_files/paper/2024/hash/0113ef4642264adc2e6924a3cbbdf532-Abstract-Conference.html  
[15] TimeXer official repo: https://github.com/thuml/TimeXer  
[16] N-HiTS AAAI article page: https://ojs.aaai.org/index.php/AAAI/article/view/25854  
[17] N-HiTS official repo: https://github.com/cchallu/n-hits  
[18] DLinear / LTSF-Linear official repo: https://github.com/cure-lab/LTSF-Linear  
[19] Time Series Library (TSLib): https://github.com/thuml/Time-Series-Library  
[20] Nixtla datasetsforecast LongHorizon source: https://github.com/Nixtla/datasetsforecast/blob/main/datasetsforecast/long_horizon.py  
[21] Nixtla NeuralForecast repo: https://github.com/Nixtla/neuralforecast  
[22] TFB arXiv: https://arxiv.org/abs/2403.20150  
[23] Accuracy Law arXiv: https://arxiv.org/abs/2510.02729

---

## 13. Open issues and limitations

This cleaned English version replaces the old chat-specific citation tokens with stable public links and a numbered reference list. The main remaining caveat is that “current SOTA” is not a single stable title in time-series forecasting. The defensible interpretation is: these are high-impact and/or leaderboard-relevant methods for the legacy LTSF and exogenous-forecasting settings as of the checked sources.

Two details should be verified immediately before formal paper submission:

1. The exact commit SHA of each repository.
2. The exact license file of any repo that does not clearly show a license on the repository page.

The most practical first step remains:

> Start with `SE-TimesNet` and `SE-NBEATS` under TSLib Look-Back-96, then evaluate `SE-NBEATSx` under the official N-BEATSx EPF protocol. Use MSE/MAE in the main tables, and place structural-entropy diagnostics, variance analysis, and exogenous-variable ablations in secondary tables.
