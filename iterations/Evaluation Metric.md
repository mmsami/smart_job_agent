# Evaluation Metric

Created by: Kyoungmi kwon
PostingDate: April 29, 2026 12:30 PM
Category: Evaluation
Last edited by: Kyoungmi kwon
Last updated time: April 30, 2026 11:13 PM


| Metric | C1 | C2 | C3 | C4 | What it Measures |
| --- | --- | --- | --- | --- | --- |
| **Relevance** | O | O | O | O | How relevant the recommended job postings are to the given CV |
| **Diversity** | O | O | O | O | How diverse the recommended job postings are |
| **Extraction Accuracy** | X | O | O | O | How accurately extracts key facts from each job posting.  |
| **Explainability** | X | O | O | O | How well explains why each job posting matches the given CV |

> Extraction Accuracy and Explainability are not applicable to C1 as these metrics were introduced by the development of LLM-based agents, and do not exist in a BM25-only pipeline.
> 

> Relevance and Diversity are evaluated on the output of Agent 3 (Reranker); Extraction Accuracy and Explainability are evaluated on the output of Agent 4 (Analyst).
> 

### Relevance

How relevant the recommended job postings are to the given CV

**Scoring:**

- Each posting → RELEVANT / NOT RELEVANT
- Overall Relevance Score = # RELEVANT postings / 10

Score range : 0 ~ 1  (A higher score indicates better relevance.)

**Metric :** 

- **Mandatory Criteria** (one failure → NOT RELEVANT):
    
    
    | # | Criteria |
    | --- | --- |
    | M1 | Job function is aligned with the CV's professional domain |
    | M2 | Job location is compatible with the CV's location preference |
    | M3 | Experience level is not severely mismatched with the CV |
- **Recommended Criteria** (≥ 75% must be met → RELEVANT):
    
    
    | # | Criteria | input |
    | --- | --- | --- |
    | R1 | Core skills overlap with the CV (≥ 2 skills) | 1/0 |
    | R2 | Seniority level is within a reasonable range of the CV | 1/0 |
    | R3 | Work type matches the CV preference (full-time / part-time) | 1/0 |
    | R4 | Industry sector is related to the CV's background | 1/0 |

### Diversity

How diverse the recommended job postings are

Diversity Score = # Unique Companies / 10

Score range : 0 ~ 1  (A higher score indicates diversity.)

| # | Criteria | input |
| --- | --- | --- |
| D1 | Unique companies in the top 10 results | 0~ 10 |

### **Extraction Accuracy - Agent 4**

How accurately extracts key facts from each job posting. 

Scoring: 

- Single-value Accuracy = (AS1+AS2+AS3+AS4) / 4
- Multi-value F1 (Required skills) = 2 × (Precision × Recall) / (Precision + Recall)
    - Precision = AM2 / AM3
    - Recall = AM2 / AM1

Score range : 0 ~ 1 (A higher score indicates more accurate and complete skill extraction.)

F1 (Required skills)

| Score | Interpretation |
| --- | --- |
| 0.9 ~ 1.0 | Near-perfect extraction |
| 0.7 ~ 0.9 | Good |
| 0.5 ~ 0.7 | Moderate — some missing or incorrectly extracted skills |
| < 0.5 | Poor — significant missing or incorrectly extracted skills |

> Note: F1 balances Precision and Recall. 
Low Precision indicates hallucinated skills 
        (extracted but not in the posting). 
Low Recall indicates missed skills 
        (in the posting but not extracted).
> 

Metric : 

- Single-value fields : — factual **correctness**
Measures whether **key factual fields** from the job posting are correctly extracted.
    
    
    | # | Criteria | Input |
    | --- | --- | --- |
    | AS1 | Job title is correctly extracted | 1/0 |
    | AS2 | Company is correctly extracted | 1/0 |
    | AS3 | Location is correctly extracted | 1/0 |
    | AS4 | Work type is correctly extracted | 1/0 |
- **Multi-value fields:**  — completeness + precision
****Measures how completely and accurately **skills** are extracted from the job posting.
    
    
    | # | Criteria | Input |
    | --- | --- | --- |
    | AM1 | How many required skills are in the original job posting?
    (ground truth - you should know answer by judge) | 0~10 |
    | AM2 | How many of them were correctly extracted? | 0~10 |
    | AM3 | How many skills were extracted in total? | 0~10 |
    
    **Data Constraints**
    
    | Condition | Reason |
    | --- | --- |
    | AM2 ≤ AM1 | Correctly extracted ≤ actual count |
    | AM2 ≤ AM3 | Correctly extracted ≤ total extracted count |

### Explainability - Agent 4

How well explains why each job posting matches the given CV

Scoring: 
Average score over 10 postings per CV

Metric:

| # | Description | Input |
| --- | --- | --- |
| EX | Explainability score | 1~5 |
- Explainability score
    
    
    | Score | Criteria |
    | --- | --- |
    | 1 | Generic explanation with no reference to the CV or job posting |
    | 2 | Mentions either the CV or the job posting, but not both |
    | 3 | Mentions both the CV and the job posting, but the connection is weak |
    | 4 | Clearly connects specific CV skills/experience to job requirements |
    | 5 | Clearly connects specific CV skills/experience to job requirements, and identifies skill gaps |