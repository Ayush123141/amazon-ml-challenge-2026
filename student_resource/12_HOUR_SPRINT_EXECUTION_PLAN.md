# ⏱️ 12-Hour Sprint Execution Plan: Amazon ML Challenge 2026

**Target Goal:** Maximize Macro $F_{0.5}$ from **0.6459 ──► 0.85+**, generate full 2.2M predictions, validate formatting, and package the final submission before the deadline.

---

## 📍 1. Current State of the Project

| Stage | What is Done | Performance / Status | Bottleneck / Next Action |
| :--- | :--- | :--- | :--- |
| **M1 Normalization** | Unicode NFC/NFKC, casefolding, punctuation stripping, address/name standardizer. | ✅ Complete & Verified | None |
| **M2 Blocking** | Exact, Casefold, House number, Rare token indexing on 100K sample. | 49.7% Candidate Recall<br>**Oracle Ceiling: 0.6759** | ⚠️ **Primary bottleneck:** 50.3% of true matches missed. |
| **M3 Classifier** | Logistic Regression & HistGradientBoosting with 37 features. | **Macro $F_{0.5}$: 0.6459**<br>Precision: 97.45% | Model is already strong on given candidates; needs better candidate recall. |
| **M4 Diagnosis** | Tested legal suffix stripping + token sorting on 173k misses. | **+55,690 missing pairs recovered (+32.0%)** | Ready to integrate into high-recall candidate generator. |

---

## 🕒 2. 12-Hour Master Timeline

```
 Hour 0 – 2   ──► Step 1: Multi-Index Candidate Generation (Target Oracle: >0.88 - 0.92)
 Hour 2 – 4   ──► Step 2: Feature Matrix Extraction (N-grams, Jaro-Winkler, Levenshtein, Tokens)
 Hour 4 – 6   ──► Step 3: Train LightGBM / CatBoost Classifier & Threshold Optimization
 Hour 6 – 9   ──► Step 4: Scale to Full 2.2M Test Set on AWS EC2 (c6i.2xlarge / m6i.2xlarge)
 Hour 9 – 10  ──► Step 5: Format Validation (utils/validate_submission.py) & Leaderboard Upload
 Hour 10 – 12 ──► Step 6: Documentation Write-up (Documentation_template.md) & Final Zip Packaging
```

---

## 💻 3. Local (12 GB RAM) vs. AWS Cloud: What is Best?

### ⚠️ Why AWS EC2 is STRONGLY Recommended for this 12-Hour Sprint:
- **Local Laptop (12 GB RAM):** Running candidate generation and feature extraction for 2.2M records locally will consume all 12 GB RAM, causing severe disk swapping, laptop freezing, and taking **4 to 6+ hours**.
- **AWS EC2 (`m6i.2xlarge` with 32 GB RAM / 8 vCPUs):** Runs the entire 2.2M pipeline in **under 20–30 minutes**.
- **Cost:** Costs **$0.00 out of pocket** (uses ~$1.00 of your free promotional credits).

---

## 👥 4. Team Task Allocation & GitHub Workflow

If you have **3 to 4 teammates and multiple AWS accounts**, you can divide and conquer in parallel:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            TEAM DIVISION OF LABOR                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  🧑‍💻 Member 1 (Lead / Modeling):                                            │
│   • Build high-recall candidate generator (M4 multi-index).                 │
│   • Train LightGBM & optimize F0.5 decision threshold on validation set.    │
│                                                                             │
│  🧑‍💻 Member 2 (AWS Master / Inference):                                      │
│   • Launch EC2 (m6i.2xlarge), pull GitHub repo, install requirements.       │
│   • Run full-scale test set candidate generation & model scoring.           │
│                                                                             │
│  🧑‍💻 Member 3 (Validation & Leaderboard):                                    │
│   • Run `utils/validate_submission.py` to ensure zero format rejections.    │
│   • Upload `matching_results.tsv` to live leaderboard & track public score. │
│                                                                             │
│  🧑‍💻 Member 4 (Documentation & Packaging):                                  │
│   • Fill in `Documentation_template.md` (methodology, architecture, params).│
│   • Prepare the final `<team_name>_submission.zip` containing code & output.│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### GitHub Setup Steps (5 Minutes):
1. **Create a Private GitHub Repo:** `amazon-ml-challenge-2026`
2. **Push codebase:**
   ```bash
   git init
   git add student_resource/src/ student_resource/m3_work/ student_resource/utils/
   git commit -m "feat: high-performance entity resolution pipeline"
   git branch -M main
   git remote add origin https://github.com/YOUR_USER/amazon-ml-challenge-2026.git
   git push -u origin main
   ```
3. Teammates and AWS EC2 instances simply run `git pull` to get the latest code.

---

## 🚀 5. Actionable Next Step: Running the High-Recall Candidate Generator

Let's execute the candidate generator script now:
```bash
python student_resource/m3_work/scripts/test_m4_candidate_generation.py
```
This will output the new **Pair Recall** and **Macro F0.5 Oracle Ceiling** immediately.
