# Data Dictionary & Brief Reconciliation

**Dataset:** `ai_job_market_insights.csv` (AI-Powered Job Market Insights, Kaggle benchmark)
**Shape:** 500 records × 10 fields · **Missing values:** 0 · **Duplicate records:** 0

## 1. Observed schema

| Field | Observed type | Levels / range | Role |
|---|---|---|---|
| `Job_Title` | Nominal categorical | 10 titles: AI Researcher, Cybersecurity Analyst, Data Scientist, HR Manager, Marketing Specialist, Operations Manager, Product Manager, Sales Manager, Software Engineer, UX Designer | Feature; unit of analysis for career mapping |
| `Industry` | Nominal categorical | 10 industries: Education, Energy, Entertainment, Finance, Healthcare, Manufacturing, Retail, Technology, Telecommunications, Transportation | Feature |
| `Company_Size` | Ordinal categorical | Small < Medium < Large | Feature |
| `Location` | Nominal categorical | 10 cities: Berlin, Dubai, London, New York, Paris, San Francisco, Singapore, Sydney, Tokyo, Toronto | Feature |
| `AI_Adoption_Level` | Ordinal categorical | Low < Medium < High | Feature |
| `Automation_Risk` | Ordinal categorical | Low / Medium / High | **Target** (modelled as 3-class) |
| `Required_Skills` | Nominal categorical (single token) | 10 skills: Communication, Cybersecurity, Data Analysis, JavaScript, Machine Learning, Marketing, Project Management, Python, Sales, UX/UI Design | Feature; basis of skill adjacency |
| `Salary_USD` | Continuous | 31,970 – 155,210 (mean 91,222; SD 20,504; 5 IQR outliers) | Feature; role-level salary signal for career mapping |
| `Remote_Friendly` | Binary | Yes / No | Feature |
| `Job_Growth_Projection` | Ordinal categorical | Decline < Stable < Growth | Feature; market-demand signal in recommendations |

## 2. Where the data differs from the case-study brief

| Field | Brief says | Data actually contains | Consequence for our approach |
|---|---|---|---|
| `AI_Adoption_Level` | Normalized numeric index (0.0–1.0) | Three ordinal categories (Low, Medium, High) | Ordinal-encoded 0/1/2; no continuous adoption gradient available |
| `Required_Skills` | Token list of competencies | Exactly **one** skill per record | Skill profiles must be built at the **job-title level** (aggregating records), not per record |
| `Job_Growth_Projection` | Numeric 5-year % growth | Three categories (Decline, Stable, Growth) | Ordinal-encoded; growth measured as the *share* of a title's postings marked Growth |
| Job titles | Examples include Data Analyst and BI Developer | Neither title is present | The scenario's most affected group (junior analysts) is not directly represented; `Data Analysis` is available as a skill category and
`Data Scientist` is present as a related analytics occupation, but neither
is treated as a substitute for the missing job titles. |

## 3. Plausibility observations
- Several skill–title combinations are atypical (for example,
  Cybersecurity Analyst requiring UX/UI Design; Sales Manager requiring
  JavaScript). Together with the weak pairwise associations observed in
  the EDA, this indicates limited occupational structure in the supplied
  dataset; it does not establish that the variables were generated
  independently.
- Target classes are near-balanced (see EDA), which is unusual for real labour-market risk data and is consistent with the same interpretation.
- These observations are examined empirically in Sections 4–6 of
  `notebooks/01_eda.ipynb` and carried forward as limitations in the
  modelling and career-mapping stages.
