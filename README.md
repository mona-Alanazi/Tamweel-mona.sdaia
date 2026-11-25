# Tamweel Lite: Predicting 90-Day Default

**Training program: Advanced ML Application · [SDAIA Academy](https://github.com/SDAIAAcademy)**

**SDA-DSC-211 · Five-part machine learning project · Free CPU on Google Colab**


**Project Builder: MONA AlAnazi - منى العنزي** 


**Email : monzi@moe.gov.sa**


**Dates: 22–25 Nov 2026**

**[GitHub Repo:](https://github.com/mona-Alanazi/Tamweel-mona.sdaia/blob/main)**


**[Colab NoteBook:](https://colab.research.google.com/drive/13ZRqlS-zzJK-eEtzIi0dCqGWjDV3_z8g?usp=sharing)**

**Final self-check (Notebook 99):** `SELF_CHECK_COMPLETE` · files to fix: 0 · manual notes: 0


This repo is my five-part journey building a credit-risk model for **Tamweel Lite**, a fictional financing dataset. The goal sounds simple: for every financing request, estimate the chance that the customer defaults within 90 days. Doing it *honestly* is the hard part. That means no data leakage, no peeking at the future, a decision threshold that respects real review capacity, and probabilities I can actually trust.

Each part built on the one before:

> **Compare models → validate honestly → choose a threshold → explain and calibrate → pick the final model and submit**

| Part | Topic | Date |
|---|---|---|
| Part 1 | Baseline vs Boosting | 22 Nov 2026 |
| Part 2 | Validate Honestly Before Tuning | 23 Nov 2026 |
| Part 3 | Choose a Threshold You Can Defend | 24 Nov 2026 |
| Part 4 | Explain the Model and Test Its Probabilities | 25 Nov 2026 |
| Part 5 | Final Model and Submission | 25 Nov 2026 |

> **Note:** All data are synthetic. This is a training exercise. It is not a real lending model and must never be used for real financing decisions.

---

## Quick facts

| | |
|---|---|
| **Target** | `default_within_90d` = 1 if a (synthetic) default happens within 90 days after the application |
| **Training data** | 10,000 requests · 22 predictors · default rate **7.89%** (789 / 10,000) |
| **Challenge data** | 2,500 later requests from new customers, without labels |
| **Final model** | **Logistic Regression** + sigmoid calibration (`KEEP SINGLE`) |
| **Best honest AP (Part 5)** | **0.392** (fold SD 0.030) |
| **Review threshold** | calibrated probability ≥ **0.1223**, capped at **12%** of each batch |
| **Challenge flags** | **300 / 2,500** sent to review (30 removed by the cap) |
| **Environment** | Python 3.13 · CPU only · seed = 211 · no GPU, no API keys |
| **Program** | Advanced ML Application · [SDAIA Academy](https://github.com/SDAIAAcademy) |

---

## Daily answers | الإجابات اليومية

### Day 1 | اليوم الأول — Part 1 · 22 Nov 2026
**أي نموذج اخترت، وما الدليل الأقوى من نتائجك؟**

اخترت **Logistic Regression**. على comparison set فيها 2,000 طلب، حقق أعلى ROC-AUC (**0.8213**) مقابل 0.8124 لـ XGBoost و0.8138 لـ LightGBM. والـ average precision كان قريبًا جدًا من الأفضل (0.3258 مقابل 0.3338 لـ XGBoost). كذلك هو أسرع بحوالي 10 مرات وأسهل في التفسير، فالـ boosting لم يقدم تحسنًا يبرر تعقيده. لكن هذا الـ split عشوائي وفيه 1,226 عميلًا مشتركًا، فالنتيجة أولية وتحتاج تحققًا أصدق.

### Day 2 | اليوم الثاني — Part 2 · 23 Nov 2026
**ما التسرب الذي اكتشفته، وكيف منعه تصميم التحقق؟**

اكتشفت **target leakage** في `days_past_due_60` و`collection_calls`، لأن معلوماتهما **after_application** ولا تكون متاحة وقت القرار. معهما حقق النموذج AP = **0.9988**، وبعد حذفهما نزل إلى **0.3110**، أي أن التسرب وحده صنع فجوة 0.688. ومنع التصميم التسرب بأربع طرق: **forward time folds**، وقاعدة **90-day label maturity** (application_date + 90 < validation start)، و**customer purge** (صفر عملاء مشتركين في كل fold)، و**training-only imputation**. وأجريت بحث Optuna على reserved cohort منفصلة. النتيجة الصادقة: AP = **0.3153 ± 0.0426**.

### Day 3 | اليوم الثالث — Part 3 · 24 Nov 2026
**لماذا لا تكون العتبة الثابتة 0.5 مناسبة دائمًا لسياسة المراجعة؟**

لأن 0.5 تفترض أن خطأ FN وخطأ FP متساويان، وأن الدرجات معايرة، وأنه لا يوجد حد للسعة، والثلاثة غير صحيحة هنا. خطأ FN يكلف 10 وحدات وFP يكلف وحدة واحدة. كما أن الـ class weighting يغيّر مستوى الدرجات، فعند 0.5 رفع النموذج weighted إشارة على **1,005** طلبات (**19.9%**)، وهذا يتجاوز سقف 12%. لذلك اخترت العتبة **0.6583**: هي أقل خسارة ضمن السعة في كل فترة، وتعطي 526 إشارة (10.4%) وrecall = 40.9% وخسارة 2,639 وحدة.

### Day 4 | اليوم الرابع — Part 4 · 25 Nov 2026
**ما أهم خاصية في تفسيرك، ولماذا لا يثبت ذلك السببية؟**

أهم خاصية هي **`bureau_score`**، وكانت الأولى في الطريقتين. خلطها أدى إلى أكبر انخفاض في AP (**0.127**)، ولها أعلى mean |SHAP| (**0.904 log-odds**)، وكلما ارتفعت قلّ الخطر. لكن هذا لا يثبت السببية: SHAP وpermutation يصفان كيف يستخدم *النموذج* الخصائص، لا ما يسبب التعثر في الواقع. كما أن الخصائص مترابطة وقد تتقاسم الإشارة، ولم نُجرِ أي تجربة أو تدخل، والبيانات اصطناعية.

### Day 5 | اليوم الخامس — Part 5 · 25 Nov 2026
**لماذا اخترت النموذج الفردي أو التجميع في النسخة النهائية؟**

اخترت النموذج الفردي **Logistic Regression** (`KEEP SINGLE`). حقق أعلى mean AP = **0.392** (fold SD = 0.030) على nested forward OOF، أعلى من XGBoost (0.353) وLightGBM (0.345) وكل التجميعات: Weighted (0.389) وStack (0.383) وEqual (0.372). لم يتجاوز أي تجميع شرط الـ Worth-It Gate (أن يتفوق بأكثر من fold SD واحد)، فالتجميع يضيف تعقيدًا بلا فائدة.

---

## The data

| File | What's inside | How I used it |
|---|---|---|
| `tamweel_train.csv` | 10,000 requests (2022–2024), 22 features + target | Training and validation |
| `tamweel_dirty.csv` | Same rows plus 2 leaky columns | Part 2 leakage audit only |
| `tamweel_challenge.csv` | 2,500 requests (2025), no labels | Final predictions only |
| `feature_dictionary.csv` | Meaning, unit and `available_at` for every column | Deciding what is allowed in the model |

**Features at a glance:** age, income, loan amount, tenor, bureau score, DTI, employment length, recent inquiries, prior defaults, obligations, savings, salary transfer, credit history, utilization, open accounts, residence years, plus region and employment flags.

**Never used as inputs:** IDs, dates, the target, and the two post-outcome columns (`days_past_due_60`, `collection_calls`).

---

## Part 1: Baseline vs Boosting

**Date:** 22 Nov 2026

**The question:** does a fancy boosting model beat a simple Logistic Regression?

I trained three models on 8,000 rows and compared them on 2,000 held-out rows. For the boosting models, the tree count was picked with early stopping on an inner split, so the comparison rows were never used for tuning.

| Model | ROC-AUC | Average Precision | Train time | Trees |
|---|---:|---:|---:|---:|
| **Logistic Regression** | **0.8213** | 0.3258 | **0.03 s** | n/a |
| XGBoost | 0.8124 | **0.3338** | 0.34 s | 61 |
| LightGBM | 0.8138 | 0.3248 | 0.24 s | 41 |

**Part 1 learning curves:**
<img width="1453" height="493" alt="day1_learning_curves" src="https://github.com/user-attachments/assets/1979a697-8e44-41a9-987e-948a757a6fd6" />


**Part 1 ROC and PR curves:**
<img width="1453" height="541" alt="day1_roc_pr" src="https://github.com/user-attachments/assets/c4bd9e6c-117b-4c94-b9f1-191535027827" />

**Takeaway:** all three are basically tied. The simplest model wins on AUC and speed. But this split was random and had **1,226 customers on both sides**, so the numbers are probably too optimistic. That's exactly what Part 2 fixes.

---

## Part 2: Validate Honestly Before Tuning

**Date:** 23 Nov 2026

**The question:** how much of my score is real, and how much is leakage?

### Leakage audit
| Column | `available_at` | Action |
|---|---|---|
| `days_past_due_60` | after_application | Dropped (post-outcome) |
| `collection_calls` | after_application | Dropped (post-outcome) |
| 22 predictors | application_date | Kept |
| IDs / date / target | metadata | Excluded |

Duplicate events found: **0**.

### Honest forward folds
Each fold trains only on the past, only on labels that have matured (90 days), and with validation customers removed.

| Fold | Validation window | Train rows | Validation rows | Validation positives | Shared customers |
|---|---|---:|---:|---:|---:|
| 1 | 2023-07 → 2023-12 | 3,223 | 1,632 | 110 | **0** |
| 2 | 2024-01 → 2024-06 | 4,460 | 1,674 | 135 | **0** |
| 3 | 2024-07 → 2024-12 | 5,731 | 1,733 | 139 | **0** |



**Fold sizes Plot :**
<img width="1093" height="445" alt="day2_fold_sizes" src="https://github.com/user-attachments/assets/1c3781d8-2cee-4bd5-aecc-c642c9afef54" />

### Bounded Optuna search
The search ran on a separate cohort of 1,935 requests (176 positives), with 8 of 8 trials completed in about 4 seconds.
Best search AP = **0.333**, with `learning_rate=0.085`, `num_leaves=31`, `min_child_samples=40`.

**Optuna search:**
<img width="1093" height="433" alt="day2_search" src="https://github.com/user-attachments/assets/2e6a504c-ae23-4b5a-b7c6-8f3702e6597f" />

### What leakage does to your score
| Protocol | Rows | ROC-AUC | AP (mean ± SD) |
|---|---:|---:|---:|
| Leaky random (unsafe) | 10,000 | 0.9999 | **0.9988** ± 0.0014 |
| Clean random (still unsafe) | 10,000 | 0.8010 | 0.3110 ± 0.0293 |
| Honest, fixed 80 trees | 5,039 | 0.7976 | **0.3153** ± 0.0426 |
| Honest, tuned | 5,039 | 0.7855 | 0.3133 ± 0.0259 |

**Validation comparison:**
<img width="1453" height="565" alt="day2_validation_comparison" src="https://github.com/user-attachments/assets/6560c4e3-7048-4f62-8794-32e27386550e" />

**Takeaway:** the leaky model looked *perfect*, which is the red flag. Once honest, AP sits around **0.31**. Tuning didn't really help, which is a useful finding in itself.

---

## Part 3: Choose a Threshold You Can Defend

**Date:** 24 Nov 2026

**The question:** at what score do we send a request to review?

**The rules:** a missed default (FN) costs **10 units** and a false alarm (FP) costs **1 unit**. Flags can't exceed **12% of requests in any period**.

### Handling class imbalance (all evaluated at 0.5)
| Strategy | ROC-AUC | AP | Accuracy | Recall | Flags | Loss | Fits capacity? |
|---|---:|---:|---:|---:|---:|---:|:---:|
| Unweighted | 0.7979 | 0.3091 | 92.6% | 12.5% | 87 | 3,399 | Yes |
| **Weighted** | 0.7969 | 0.3100 | 81.3% | 57.8% | 1,005 | 2,403 | No |
| Oversampled | 0.7979 | 0.3159 | 80.7% | 59.9% | 1,047 | 2,357 | No |
| *Flag nobody* | n/a | n/a | **92.4%** | **0%** | 0 | 3,840 | Yes |

"Flag nobody" scores 92.4% accuracy and catches zero defaults. That's why accuracy alone is misleading here.

**Part 3 ROC and PR:**
<img width="1393" height="529" alt="day3_roc_pr" src="https://github.com/user-attachments/assets/5b349bff-e730-4c19-b201-4ff700d60285" />

### Picking the operating point (weighted model)
| Rule | Threshold | Flags | Recall | Precision | Loss | Fits capacity? |
|---|---:|---:|---:|---:|---:|:---:|
| Default 0.5 | 0.5000 | 1,005 (19.9%) | 57.8% | 22.1% | 2,403 | No |
| Min loss, no limit | 0.4486 | 1,141 (22.6%) | 64.1% | 21.6% | 2,275 | No |
| **Min loss within capacity** | **0.6583** | **526 (10.4%)** | 40.9% | 29.8% | 2,639 | **Yes** |

**Cost curve:**
<img width="1213" height="793" alt="cost_curve" src="https://github.com/user-attachments/assets/f8aa07ca-1408-4c22-9479-3dd0debe4ebe" />

### Capacity check, period by period
| Period | Requests | Capacity (12%) | Flags | Within capacity? |
|---|---:|---:|---:|:---:|
| 1 | 1,632 | 195 | 137 | Yes |
| 2 | 1,674 | 200 | 183 | Yes |
| 3 | 1,733 | 207 | 206 | Yes (very tight) |

**Sensitivity:** with FN cost set to 8, 10 or 12, the threshold stays at **0.6583**, because capacity is the binding constraint.
**Regions:** the false-positive rate ranges from 7.64% to 8.29%, a gap of **0.65 percentage points**. This is a descriptive check only, not a fairness certificate.

**Capacity and regions:**
<img width="1333" height="493" alt="day3_capacity_regions" src="https://github.com/user-attachments/assets/0f3a3d08-6968-42ba-8a5a-fde616932cc1" />

**Takeaway:** respecting capacity costs **+236 loss units** compared with 0.5, but 0.5 simply isn't doable in practice.

---

## Part 4: Explain the Model and Test Its Probabilities

**Date:** 25 Nov 2026

**The questions:** what is the model actually using, and can its scores be read as probabilities?

| Role | Period | Rows | Positives | Used for |
|---|---|---:|---:|---|
| fit | up to 2023-04 | 2,516 | 217 | training |
| calibration | Jul–Sep 2023 | 584 | 40 | sigmoid only |
| policy | Jan–Mar 2024 | 589 | 55 | threshold only |
| evaluation | Jul–Dec 2024 | 1,733 | 139 | measuring |

### What drives the model
| Rank | Feature | Permutation AP drop | Mean \|SHAP\| (log-odds) |
|---:|---|---:|---:|
| 1 | **bureau_score** | **0.127** | **0.904** |
| 2 | dti | 0.069 | 0.544 |
| 3 | loan_amount_sar | 0.014 | 0.349 |
| 4 | savings_balance_sar | 0.009 | 0.217 |
| 5 | prior_defaults | 0.007 | 0.183 |

**Permutation importance:**

<img width="1573" height="613" alt="permutation_importance" src="https://github.com/user-attachments/assets/fab843c9-c46e-4161-b652-b3ae0469b53c" />


**SHAP beeswarm:**

<img width="1279" height="705" alt="shap_beeswarm" src="https://github.com/user-attachments/assets/9ac7ea4d-57f4-4802-be92-e0d1f04722e8" />

### One request up close: `TR-009585`
This is the highest-scoring request in the SHAP sample. Its raw score is **0.903**, but after calibration it's only **0.480**.

| Reason | Value | SHAP (log-odds) |
|---|---:|---:|
| Low bureau score | 497 | +2.27 |
| High DTI | 1.28 | +1.20 |
| Loan amount | 93,437 SAR | +0.20 |

Changing `bureau_score` by ±1 left the prediction and the top 3 reasons unchanged.

**SHAP waterfall**

<img width="1183" height="769" alt="shap_waterfall" src="https://github.com/user-attachments/assets/5c5c7a52-64bb-4883-8aa2-20a08ad3e8ac" />

### Calibration (measured on a separate period)
| | ROC-AUC | AP | Brier (lower is better) | ECE (lower is better) |
|---|---:|---:|---:|---:|
| Raw (weighted) | 0.7708 | 0.2587 | 0.1130 | 0.1469 |
| **Sigmoid** | 0.7708 | 0.2587 | **0.0671** | **0.0225** |

Ranking stays the same (the mapping is monotone), but the probabilities became much more honest.

**Reliability curve**

<img width="1213" height="973" alt="reliability_curve" src="https://github.com/user-attachments/assets/999997bf-87bb-4690-81e0-cfab42929664" />

**Stability:** 200 customer-cluster bootstrap replicates gave an AP 95% interval of [0.197, 0.338]. The Brier change interval was [−0.054, −0.037], negative in every replicate.

**Stability summary**

<img width="1453" height="493" alt="stability_summary" src="https://github.com/user-attachments/assets/16dc79a7-870b-4afa-a326-37241dc9eb01" />

### Review zone vs capacity
The threshold was chosen on the policy role (raw 0.588, calibrated 0.173), with a ±0.02 diagnostic band.

| Quarter | Capacity | Risk flags | Flags + near threshold | Status |
|---|---:|---:|---:|---|
| 2024Q3 | 100 | 97 | 109 | over capacity once the band is added |
| 2024Q4 | 107 | 109 | 122 | over capacity even without the band |

Result: **`CAPACITY_REVIEW_REQUIRED`.** I didn't retune on evaluation, because a new policy needs new evidence.

**Review zone**

<img width="1453" height="553" alt="review_zone" src="https://github.com/user-attachments/assets/173cd51b-5957-4305-8960-7a4e5c006fbe" />

---

## Part 5: Final Model and Submission

**Date:** 25 Nov 2026

**The question:** is an ensemble worth the extra complexity?

### Worth-It Gate (nested forward OOF, 2,155 rows, 3 folds)
| Candidate | Mean AP | Fold SD | Brier | ECE | Lift vs best single | Passes? |
|---|---:|---:|---:|---:|---:|:---:|
| LightGBM | 0.345 | 0.043 | 0.0661 | 0.0231 | −0.046 | No |
| XGBoost | 0.353 | 0.029 | 0.0657 | 0.0228 | −0.039 | No |
| **Logistic** | **0.392** | 0.030 | **0.0633** | 0.0188 | n/a | best single |
| Equal blend | 0.372 | 0.033 | 0.0644 | 0.0204 | −0.020 | No |
| Weighted blend | 0.389 | 0.029 | 0.0633 | 0.0177 | −0.002 | No |
| Stack | 0.383 | 0.029 | 0.0660 | 0.0311 | −0.009 | No |

**Decision: `KEEP SINGLE` → Logistic Regression.**

**Model diversity**

<img width="1313" height="553" alt="day5_diversity" src="https://github.com/user-attachments/assets/d669d6f3-7d7b-42c9-a4c8-c42fe478e84f" />

**Ensemble comparison**

<img width="1573" height="613" alt="day5_ensemble_comparison" src="https://github.com/user-attachments/assets/8f5f82d8-5fd1-4b40-aeb4-506e5c971172" />

### Final policy
| | Value |
|---|---|
| OOF threshold (raw) | 0.1689 |
| Calibrated threshold | **0.1223** |
| OOF recall / precision | 46.9% / 34.3% |
| OOF loss (10·FN + FP) | 1,111 units |
| Most loaded period | 11.7% (within capacity) |
| Regional FPR | 6.31% (eastern) to 10.29% (western), a gap of about 4 pp that needs review |

**Policy and regions**

<img width="1453" height="553" alt="day5_policy_regions" src="https://github.com/user-attachments/assets/d5242a03-0f9f-47ae-82f8-67f4950b81eb" />

**Calibration fit**

<img width="1453" height="589" alt="day5_calibration_fit" src="https://github.com/user-attachments/assets/c57d244b-cada-4535-9210-7f610931641c" />

### Challenge batch (2,500 unlabeled requests)
| Above threshold | Capacity (12%) | Final flags | Removed by cap |
|---:|---:|---:|---:|
| 330 | 300 | **300** | 30 |

**Challenge capacity**

<img width="1333" height="493" alt="day5_challenge_capacity" src="https://github.com/user-attachments/assets/8c7cc139-bb16-4236-8c24-aa39d10f6e7e" />

The submission was replayed from the exported model and matched exactly (`REPLAY_MATCH`).

---

## What I learned

| Lesson | Evidence from my run |
|---|---|
| A perfect score is a warning sign | Leaky AP 0.9988 vs honest 0.3153 |
| Simple can win | Logistic beat every ensemble |
| Accuracy lies with imbalanced data | "Flag nobody" = 92.4% accuracy, 0% recall |
| 0.5 is just a number | It flagged 19.9% of requests vs a 12% capacity |
| Weighted scores are not probabilities | Raw 0.903 became 0.480 after calibration |
| Explanations are not causes | `bureau_score` is the top feature, but not proven to *cause* default |

## Limitations

- The data are **synthetic**, so results don't transfer to real customers.
- Folds are related time periods, not independent experiments. SD is descriptive, not a confidence interval.
- Thresholds were chosen on OOF labels, so the reported loss is optimistic development evidence.
- The Part 4 SHAP analysis explained a weighted LightGBM, not the final Logistic model, so it should be re-checked on the final model.
- The regional checks are descriptive only, with no fairness certification.
- Capacity can fail in future periods (it already did in Part 4's evaluation), so it needs monitoring.

## Repository structure

```
Tamweel-mona.sdaia/
├── README.md                     ← this file (includes Daily answers)
├── requirements-colab.txt        ← pinned package versions
├── constraints.txt
│
├── notebooks/                    ← executed Colab notebooks
│   ├── 01_baseline_boosting.ipynb
│   ├── 02_validation_tuning.ipynb
│   ├── 03_cost_sensitive_decision.ipynb
│   ├── 04_explain_calibrate.ipynb
│   ├── 05_final_model.ipynb
│   └── 99_final_submission_check.ipynb
│
├── data/                         ← course data (SHA-256 verified)
│   ├── data_contract.json
│   ├── data_manifest.json
│   ├── tamweel_train.csv
│   ├── tamweel_dirty.csv
│   └── tamweel_challenge.csv
│
├── evidence/                     ← outputs + run records for Parts 1–4
│   ├── day1/   model comparison, predictions, split membership, curves, day1_run.json
│   ├── day2/   leakage & fold audits, Optuna results, best params, day2_run.json
│   ├── day3/   threshold metrics & sweep, OOF predictions, region audit, cost curve, day3_run.json
│   └── day4/   calibration metrics, permutation importance, SHAP, reliability, day4_run.json
│
├── artifacts/                    ← Part 5 final outputs
│   ├── final_model/              model.json · model_manifest.json
│   ├── final_metrics.json
│   ├── final_policy.json
│   ├── ensemble_comparison.csv
│   ├── day5_run.json · environment.json
│   ├── day5_oof_predictions.csv · day5_fold_scores.csv
│   ├── day5_oof_provenance.json · day5_final_provenance.json
│   └── day5_*.png                figures
│
├── reports/
│   ├── MODEL_CARD.md
│   └── ENSEMBLE_DECISION.md
│
├── submission/
│   └── submission.csv            ← 2,500 rows: application_id, probability, decision
│
├── scripts/
│   ├── inference.py
│   └── rebuild_final.py
│
└── tamweel/                      ← project package
    ├── __init__.py
    └── inference.py
```

## How to reproduce

1. Open any notebook in Google Colab and pick the free **CPU** runtime.
2. Choose **Runtime → Run all**. Setup checks package versions and file hashes automatically.
3. Keep `FAST_MODE=True` and seed `211`.
4. Results should match the tables above, give or take the last decimal.

## Final check

Open `notebooks/99_final_submission_check.ipynb` in Colab, set `PROJECT_URL` to this repository link (leave `PROJECT_ZIP` empty), and run the cells. Current result:

```
SELF_CHECK_COMPLETE | files to fix: 0 | manual notes: 0
```

---

*Educational project for SDA-DSC-211 · Advanced ML Application program by [SDAIA Academy](https://github.com/SDAIAAcademy). Synthetic data only. Not for real financing decisions.*
