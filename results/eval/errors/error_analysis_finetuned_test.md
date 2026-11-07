# Error analysis (finetuned_test)

Category counts:

- **unsupported_decision**: 129
- **decision_without_information**: 29
- **right_label_wrong_evidence**: 23
- **applicability_error**: 15
- **missed_information**: 12
- **contradiction_same_evidence_wrong_label**: 1

## Case 1: unsupported_decision  (annotation 165, sigir-201410, NCT02097186)
- Criterion (exclusion): Pregnancy
- Expert: **not applicable**, evidence []
- Model: **not excluded**, evidence []
- Raw output: `{"evidence": [], "label": "not excluded"}`
- Gold evidence text: (none)
- Predicted evidence text: (none)
- Analysis: _write your interpretation here_

## Case 2: unsupported_decision  (annotation 166, sigir-201410, NCT02097186)
- Criterion (exclusion): Significant upper limb peripheral arterial disease
- Expert: **not excluded**, evidence []
- Model: **not excluded**, evidence []
- Raw output: `{"evidence": [], "label": "not excluded"}`
- Gold evidence text: (none)
- Predicted evidence text: (none)
- Analysis: _write your interpretation here_

## Case 3: decision_without_information  (annotation 169, sigir-201410, NCT02097186)
- Criterion (exclusion): Patients with an estimated pre-operative glomerular filtration rate < 30mls/min/1.73m2
- Expert: **not enough information**, evidence []
- Model: **not excluded**, evidence []
- Raw output: `{"evidence": [], "label": "not excluded"}`
- Gold evidence text: (none)
- Predicted evidence text: (none)
- Analysis: _write your interpretation here_

## Case 4: decision_without_information  (annotation 174, sigir-201410, NCT02097186)
- Criterion (exclusion): Patients with previous axillary surgery
- Expert: **not enough information**, evidence []
- Model: **not excluded**, evidence []
- Raw output: `{"evidence": [], "label": "not excluded"}`
- Gold evidence text: (none)
- Predicted evidence text: (none)
- Analysis: _write your interpretation here_

## Case 5: right_label_wrong_evidence  (annotation 179, sigir-201410, NCT02264964)
- Criterion (exclusion): Had been performed central venous puncture or catheterization before.
- Expert: **not excluded**, evidence [0]
- Model: **not excluded**, evidence []
- Raw output: `{"evidence": [], "label": "not excluded"}`
- Gold evidence text: [0] A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.
- Predicted evidence text: (none)
- Analysis: _write your interpretation here_

## Case 6: right_label_wrong_evidence  (annotation 196, sigir-201411, NCT01723137)
- Criterion (exclusion): Less than 18 years of age
- Expert: **not excluded**, evidence [0]
- Model: **not excluded**, evidence []
- Raw output: `{"evidence": [], "label": "not excluded"}`
- Gold evidence text: [0] A 40-year-old woman with no past medical history presents to the ER with excruciating pain in her right arm that had started 1 hour prior to her admission.
- Predicted evidence text: (none)
- Analysis: _write your interpretation here_

## Case 7: applicability_error  (annotation 422, sigir-201422, NCT00236912)
- Criterion (inclusion): If female, using birth control
- Expert: **not applicable**, evidence [3]
- Model: **not enough information**, evidence []
- Raw output: `{"evidence": [], "label": "not enough information"}`
- Gold evidence text: [3] She has had no sexual partners and her menses are regular.
- Predicted evidence text: (none)
- Analysis: _write your interpretation here_

## Case 8: missed_information  (annotation 161, sigir-201410, NCT02097186)
- Criterion (inclusion): Patients undergoing elective carotid endarterectomy
- Expert: **not included**, evidence []
- Model: **not enough information**, evidence []
- Raw output: `{"evidence": [], "label": "not enough information"}`
- Gold evidence text: (none)
- Predicted evidence text: (none)
- Analysis: _write your interpretation here_

## Case 9: missed_information  (annotation 162, sigir-201410, NCT02097186)
- Criterion (inclusion): Patients undergoing open abdominal aortic aneurysm repair
- Expert: **not included**, evidence []
- Model: **not enough information**, evidence []
- Raw output: `{"evidence": [], "label": "not enough information"}`
- Gold evidence text: (none)
- Predicted evidence text: (none)
- Analysis: _write your interpretation here_

## Case 10: contradiction_same_evidence_wrong_label  (annotation 762, sigir-201513, NCT00315042)
- Criterion (inclusion): A sore and scratchy throat and/or pain on swallowing (odynophagia) together with at least 2 of the following clinical signs:
- Expert: **not enough information**, evidence [0]
- Model: **included**, evidence [0]
- Raw output: `{"evidence": [0], "label": "included"}`
- Gold evidence text: [0] A 5-year-old boy presents to the emergency department with complaints of progressively worsening dysphagia, drooling, fever and vocal changes.
- Predicted evidence text: [0] A 5-year-old boy presents to the emergency department with complaints of progressively worsening dysphagia, drooling, fever and vocal changes.
- Analysis: _write your interpretation here_
