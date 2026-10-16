# Three-Minute Faculty Demonstration Script — Step 1

**Product:** Student Performance and Academic Intervention System (SC03)  
**Order of Presentation:** Strictly follows Page 13 of the Faculty Guide  

---

### Step 1: Repository & Collaborators (15 seconds)
- **Action:** Open GitHub repository / local git log.
- **Talking Point:** *"Good morning. Here is our SC03 repository. All members are active collaborators, our branches and commit history follow atomic commit discipline, and all dependencies are pinned in `requirements.txt`."*

### Step 2: One-Sentence Problem in README.md (15 seconds)
- **Action:** Open [README.md](file:///c:/Users/Abhishek/Capstone%20Project/README.md).
- **Talking Point:** *"Our project problem is: 'Given anonymous academic indicators, estimate a risk band and suggest a supportive intervention.' The system is designed for early decision support and never automates student punishment."*

### Step 3: Project Contract & User Roles (20 seconds)
- **Action:** Open [docs/STEP-1.md](file:///c:/Users/Abhishek/Capstone%20Project/docs/STEP-1.md).
- **Talking Point:** *"The primary user is a faculty mentor. In M1, we deliver Product V1 with a score baseline and one transparent ADALINE classifier. In M2, we compare Perceptron, ADALINE, and Backpropagation with fairness audits."*

### Step 4: Data Dictionary & Assumptions (20 seconds)
- **Action:** Open [data/README.md](file:///c:/Users/Abhishek/Capstone%20Project/data/README.md).
- **Talking Point:** *"Our schema consists of 5 academic indicators: Attendance %, Continuous Assessment %, Assignment %, LMS Engagement (0–10), and Prior Performance %. All real identifiers are strictly banned. Missing values trigger an immediate row rejection."*

### Step 5: Starter Sample Breakdown (25 seconds)
- **Action:** Open [data/sample_input.csv](file:///c:/Users/Abhishek/Capstone%20Project/data/sample_input.csv).
- **Talking Point:** *"We created 20 starter rows across four partitions: normal cases like STU-001, boundary cases like STU-005, and difficult cases like STU-004 where engagement is 9/10 but assessment marks are only 48%."*

### Step 6: Run Data Validator (PASS) (20 seconds)
- **Action:** Run `python src/validate_data.py data/sample_input.csv` in terminal.
- **Talking Point:** *"Running the validator on our 20-row sample checks shape, columns, numeric bounds, and PII absence. Result: STEP 1 DATA CHECK PASSED."*

### Step 7: Run Data Validator (FAIL) (20 seconds)
- **Action:** Run `python src/validate_data.py data/invalid_cases.csv` in terminal.
- **Talking Point:** *"Testing row STU-BAD with 112% attendance and -1 engagement immediately catches all 4 errors and reports them in a clear numbered list with exit code 1."*

### Step 8: Explain Baseline Rules & Golden Case (25 seconds)
- **Action:** Open [docs/BASELINE_PSEUDOCODE.md](file:///c:/Users/Abhishek/Capstone%20Project/docs/BASELINE_PSEUDOCODE.md).
- **Talking Point:** *"Our baseline is a transparent linear weighting: 30% attendance, 30% assessment, 20% assignment, 10% engagement, and 10% prior performance. For STU-001, 0.3(92) + 0.3(84) + 0.2(88) + 0.1(80) + 0.1(81) = 86.5, which is Low risk (threshold >= 70)."*

### Step 9: Product V1 Screen Sketch & Twin Read (20 seconds)
- **Action:** Display [docs/product-v1-sketch.png](file:///c:/Users/Abhishek/Capstone%20Project/docs/product-v1-sketch.png).
- **Talking Point:** *"Here is our Product V1 wireframe. The signature element is the Twin Read: a single horizontal scale comparing baseline and neural outputs. When they disagree, the visual gap is highlighted so the mentor can make an informed human decision."*

### Step 10: Git Commit History (10 seconds)
- **Action:** Run `git log --oneline`.
- **Talking Point:** *"Here is our Git history demonstrating clean, single-purpose commits covering our repository setup, contract, schema, golden tests, validator, and wireframe."*
