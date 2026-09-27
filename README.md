TrustLend: built at HackGenix 2026 (Technosummit) by Akshay S Krishnan, Chakradeep, Tanush and Gokul. Original team repo: https://github.com/chakradeepreddy/trustlend · Live demo: https://trustlend.streamlit.app

# TrustLend

### A loan AI that knows when to ask a human.

**Live Demo:** https://trustlend.streamlit.app  
**GitHub:** https://github.com/chakradeepreddy/TrustLend

TrustLend is a safety layer on top of a loan-risk ML model. Instead of blindly trusting an AI prediction, TrustLend checks whether the prediction is close to the decision boundary and whether the applicant is unfamiliar with the population represented in the training data. If either safety check fires, the case is sent to a human reviewer. Otherwise, the original AI decision is allowed to stand.

---

## Why TrustLend?

A loan model can be confident without necessarily having enough evidence behind that confidence.

Two different failure modes matter:

1. **Borderline decision:** the predicted risk is close to the approve/deny cutoff.
2. **Unfamiliar applicant:** the applicant is far from the people represented in the training/reference data.

TrustLend handles both by adding a human-defer layer around the normal AI decision.

---

## End-to-End Architecture

```text
                    HISTORICAL LOAN DATA
                            |
                            v
                       src/data.py
                  load + clean + split
                            |
              +-------------+-------------+
              |                           |
              v                           v
        src/model.py              src/familiarity.py
        Loan-risk model             Familiarity
              |                       Check 2
              v                           |
       model.joblib                       v
              |                    familiarity.joblib
              |                           |
              +-------------+-------------+
                            |
                            v
                       src/policy.py
                  choose safety thresholds
                            |
                            v
                     thresholds.json
                            |
                            v
                      src/decide.py
                            |
                     New applicant
                            |
                            v
                  +-------------------+
                  | ML predicts risk  |
                  +---------+---------+
                            |
                 +----------+----------+
                 |                     |
                 v                     v
              Check 1               Check 2
          near risk cutoff?      unfamiliar?
                 |                     |
                 +----------+----------+
                            |
                            v
                 +-------------------+
                 | Either check fires?|
                 +---------+---------+
                           / \
                         YES  NO
                          |    |
                          v    v
                      HUMAN   AI decision
                      REVIEW   stands
                          |
                          v
                 Supabase PostgreSQL
                  /               \
            review_cases       audit_logs
```

---

# Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.12 | Core application, ML and data pipeline |
| Frontend / Web App | Streamlit | Applicant form, results, review queue, audit ledger, dashboard |
| ML | scikit-learn | Model training, calibration, nearest-neighbour familiarity |
| Main Model | HistGradientBoostingClassifier | Loan default-risk prediction |
| Calibration | CalibratedClassifierCV + isotonic calibration | Makes predicted probabilities more meaningful |
| Familiarity | NearestNeighbors + RobustScaler | Check 2: distance from reference applicants |
| Data | Pandas + NumPy | Loading, cleaning, transformation and evaluation |
| Model Persistence | Joblib | Save/load trained ML artifacts |
| Charts | Plotly | Evaluation dashboard |
| Database | PostgreSQL | Persistent review and audit data |
| Database Hosting | Supabase | Hosted PostgreSQL |
| PostgreSQL Driver | psycopg2-binary | Python database connection |
| Version Control | Git + GitHub | Source control and collaboration |
| Deployment | Streamlit Community Cloud | Public deployment |

---

# Dataset

TrustLend uses the **Give Me Some Credit** loan dataset.

The target is:

```text
SeriousDlqin2yrs
```

The dataset contains about **150,000 applicants** and includes features such as:

- Age
- Monthly income
- Debt ratio
- Revolving credit utilization
- Number of open credit lines/loans
- 30–59 day late payments
- 60–89 day late payments
- 90+ day late payments
- Number of real-estate loans/lines
- Number of dependents

The current split used by TrustLend is:

```text
Training:    86,391
Validation:  28,797
Test:        28,797
Strangers:    6,014
```

The current "stranger" population is the **top 5% of MonthlyIncome**.

---

# 1. Data Pipeline

`src/data.py` is responsible for:

- loading the dataset
- cleaning/preparing the data
- building training, validation, test and stranger groups
- generating corrupted/typo test data for evaluation

The evaluation is performed on applicants the model did not train on.

---

# 2. Loan-Risk ML Model

## Main algorithm

TrustLend uses:

```python
HistGradientBoostingClassifier(random_state=0)
```

## Probability calibration

The classifier is wrapped with:

```python
CalibratedClassifierCV(
    HistGradientBoostingClassifier(random_state=0),
    method="isotonic",
    cv=5
)
```

Calibration is important because TrustLend uses the predicted probability itself in its decision policy.

The model produces:

```text
P(default | applicant)
```

Examples:

```text
2.16% risk
11.57% risk
35.5% risk
```

These are generated by the trained model.

## Model evaluation

The integrated build reported a validation ROC-AUC of approximately **0.863**.

---

# 3. The AI Decision Cutoff

TrustLend uses these modeling assumptions:

```text
Cost of missed default = 5
Cost of wrong denial  = 1
```

For predicted default probability `p`:

```text
Approve expected loss = 5p
Deny expected loss    = 1(1-p)
```

Setting them equal:

```text
5p = 1 - p
6p = 1
p = 1/6
p ≈ 0.167
```

Therefore the cutoff is approximately:

```text
16.7%
```

The plain model behaves as:

```text
Risk < 16.7%  -> APPROVE
Risk >= 16.7% -> DENY
```

---

# 4. Check 1 — "Is the AI too close to the line?"

Check 1 looks at the model's predicted risk.

The current safety band is:

```text
8.7% to 24.7%
```

If the predicted risk falls inside that band, TrustLend sends the case to human review.

Conceptually:

```text
            16.7%
              |
8.7% ---------+--------- 24.7%
        HUMAN REVIEW ZONE
```

---

# 5. Check 2 — "Has the model seen people like this?"

Check 2 is independent of the model's risk probability.

It asks:

> How similar is this applicant to applicants represented in the training/reference population?

## Reference population

TrustLend samples up to:

```text
30,000 training applicants
```

## Preprocessing

Before distance calculation:

1. Add an `income_missing` indicator.
2. Fill missing values with training medians.
3. Apply `log1p` to:
   - MonthlyIncome
   - DebtRatio
   - RevolvingUtilizationOfUnsecuredLines
4. Apply `RobustScaler`.

## Nearest neighbours

TrustLend uses:

```python
NearestNeighbors(n_neighbors=10)
```

It finds the 10 nearest reference applicants and computes:

```text
familiarity distance =
average distance to the 10 nearest applicants
```

Smaller distance means closer to the reference population; larger distance means less familiar.

The 10 nearest reference applicants are shown in the UI for reviewer context.

---

# 6. Final TrustLend Policy

```text
                 AI risk prediction
                         |
              +----------+----------+
              |                     |
          Check 1                Check 2
          borderline?            unfamiliar?
              |                     |
              +----------+----------+
                         |
                    Either fires?
                     /         \
                   YES          NO
                    |            |
                    v            v
               HUMAN REVIEW   AI DECISION
                                 STANDS
```

TrustLend does not replace the loan model. It decides when the model should be allowed to make the decision automatically.

---

# 7. Runtime Decision Flow

For every applicant:

```text
1. User submits applicant data
2. Streamlit calls decide(applicant)
3. Saved ML model predicts default risk
4. Plain AI decision is calculated using the 16.7% cutoff
5. Check 1 evaluates distance from the cutoff
6. Check 2 calculates familiarity distance
7. If either check fires -> REVIEW
8. Otherwise -> original AI decision
9. REVIEW cases are persisted in PostgreSQL
10. Human reviewer can approve/deny
11. Final action is written to the audit history
```

No model retraining happens when the user clicks "Decide".

---

# 8. Human Review Workflow

When a case is sent to review, the reviewer sees:

- applicant information
- AI predicted risk
- original AI decision
- Check 1 status
- Check 2 status
- familiarity distance
- reasons for deferral
- nearest reference applicants
- reviewer note field

The reviewer can:

```text
APPROVE
or
DENY
```

The reviewer decision, note, actor and timestamp are stored.

---

# 9. PostgreSQL + Supabase

The deployed application uses **PostgreSQL hosted by Supabase**.

Two main tables are used:

## `review_cases`

Stores the full lifecycle of a human-review case:

- applicant data
- model decision
- predicted probability
- Check 1 result
- Check 2 result
- familiarity distance
- reasons
- status
- reviewer
- reviewer decision
- reviewer note
- timestamps

## `audit_logs`

Stores decision history for both automatic and human-reviewed decisions:

- time
- applicant
- AI decision
- risk
- TrustLend decision
- final decision
- decided by
- reasons
- note

PostgreSQL is the source of truth for operational review/audit data.

---

# 10. Build Time vs Runtime

TrustLend has two phases.

## Build time

```text
data.py
   |
   +--> model.py
   |
   +--> familiarity.py
   |
   +--> policy.py
   |
   v
artifacts/
```

This produces:

```text
model.joblib
familiarity.joblib
thresholds.json
```

Evaluation produces:

```text
results.json
```

## Runtime

The deployed app loads those saved artifacts and runs:

```text
new applicant -> decide()
```

No training happens during normal demo usage.

---

# 11. Evaluation

The Dashboard evaluates three groups.

### Normal Applicants

```text
28,797 applicants
13.5% referred to human
Plain model cost: 0.209
TrustLend cost: 0.126
```

### Strangers

```text
6,014 applicants
20.3% referred to human
Plain model cost: 0.193
TrustLend cost: 0.093
```

### Typos

```text
28,797 applicants
68.1% referred to human
Plain model cost: 0.222
TrustLend cost: 0.044
```

The current dashboard displays approximate modeled cost reductions of:

```text
Normal:    40%
Strangers: 52%
Typos:     80%
```

## What "cost" means

This is a **modeled average decision loss for the decisions the AI still makes automatically**, using the assumed mistake costs:

```text
Missed default = 5
Wrong denial   = 1
```

It is not a literal company invoice and does not include the operational cost of employing human reviewers.

---

# 12. Dashboard

The Dashboard answers:

> Does the TrustLend safety layer improve the decisions the AI makes on its own?

It contains:

1. Test-set summary
2. Normal / Stranger / Typo result cards
3. Results table
4. Cost comparison visualization
5. Cost-vs-human-referral curve

The graph compares:

- **Random**
- **Check 1 only**
- **TrustLend**

X-axis:

```text
Percentage referred to humans
```

Y-axis:

```text
Average modeled cost per automatic AI decision
```

Moving right means more cases are sent to humans. Moving down means lower modeled cost among the decisions still made automatically by the AI.

---

# 13. Web Application

### Home

- Applicant form
- Demo presets
- AI risk
- Check 1
- Check 2
- final decision
- reasons
- 10 nearest applicants

### How It Works

Explains the full TrustLend pipeline.

### Review Queue

- pending human cases
- review reasons
- reviewer note
- approve / deny
- audit ledger

### Dashboard

- evaluation metrics
- test-set results
- cost comparison
- referral/cost visualization

---

# 14. Demo Presets

The application includes cases for the major branches:

```text
Normal
    -> APPROVE

Borderline
    -> REVIEW via Check 1

Stranger
    -> REVIEW via Check 2

Typo
    -> REVIEW via Check 2

Deny
    -> DENY
```

These allow the complete decision tree to be demonstrated live.

---

# 15. Deployment

## Live application

**https://trustlend.streamlit.app**

## Deployment architecture

```text
GitHub
   |
   v
Streamlit Community Cloud
   |
   +--> Streamlit UI
   +--> Python decision engine
   +--> ML artifacts
   |
   v
Supabase PostgreSQL
   |
   +--> review_cases
   +--> audit_logs
```

Database credentials are supplied through deployment secrets and are not committed to source control.

---

# 16. Local Setup

Clone:

```bash
git clone https://github.com/chakradeepreddy/TrustLend.git
cd TrustLend
```

Install:

```bash
pip install -r requirements.txt
```

Build artifacts:

```bash
python -m scripts.build
```

Generate evaluation results:

```bash
python -m src.evaluate
```

Run:

```bash
streamlit run app/Home.py
```

For local PostgreSQL-backed operation, configure the required database secret/environment variable for the application.

---

# 17. Repository Structure

```text
TrustLend/
|
├── app/
│   ├── Home.py
│   ├── db.py
│   ├── audit.py
│   ├── sidebar.py
│   ├── style.py
│   └── pages/
│       ├── 0_How_it_works.py
│       ├── 1_Review_queue.py
│       └── 2_Dashboard.py
|
├── src/
│   ├── data.py
│   ├── model.py
│   ├── familiarity.py
│   ├── policy.py
│   ├── decide.py
│   └── evaluate.py
|
├── scripts/
│   ├── build.py
│   └── migrate_audit_csv.py
|
├── artifacts/
│   ├── model.joblib
│   ├── familiarity.joblib
│   ├── thresholds.json
│   └── results.json
|
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 18. Design Principles

### AI first, but not AI only

The model makes the normal prediction. TrustLend decides when that prediction needs additional scrutiny.

### Selective human review

Humans review cases identified as borderline or unfamiliar rather than reviewing every application.

### Evidence, not just a score

The reviewer gets reasons, familiarity distance and similar reference applicants.

### Persistent operational data

Review cases and audit history are persisted in PostgreSQL.

### Build once, serve many

Training and threshold selection happen during the build phase. Runtime uses saved artifacts.

---

# 19. Limitations

TrustLend is a hackathon prototype, not a production lending decision system.

Important limitations:

- The cost values are modeling assumptions.
- Familiarity depends on the selected reference population and preprocessing choices.
- Familiarity distance is a relative metric, not a probability of correctness.
- Evaluation results depend on the selected dataset and test constructions.
- Human review introduces operational cost/time that is not included in the dashboard's modeled decision-loss metric.
- A real-world lending system would require additional validation, governance, fairness analysis, security controls and regulatory/compliance review.

---

# 20. Core Idea

> **TrustLend does not try to make the model smarter; it makes the system know when the model should ask for help.**

---

## Live Demo

https://trustlend.streamlit.app

## GitHub

https://github.com/chakradeepreddy/TrustLend
