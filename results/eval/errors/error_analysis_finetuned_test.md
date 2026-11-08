# Error analysis (finetuned_test)

Category counts:

- **unsupported_decision**: 121
- **decision_without_information**: 29
- **right_label_wrong_evidence**: 27
- **applicability_error**: 15
- **missed_information**: 10
- **contradiction_same_evidence_wrong_label**: 2
- **polarity_error**: 1

## Case 1: unsupported_decision  (annotation 165, sigir-201410, NCT02097186)
- Criterion (exclusion): Pregnancy
- Expert: **not applicable**, evidence []
- Model: **not excluded**, evidence []
- Raw output: `{"evidence": [], "label": "not excluded"}`
- Gold evidence text: (none)
- Predicted evidence text: (none)
- Analysis: _write your interpretation here_

## Case 2: unsupported_decision  (annotation 167, sigir-201410, NCT02097186)
- Criterion (exclusion): Previous history of upper limb deep vein thrombosis
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

## Case 5: right_label_wrong_evidence  (annotation 166, sigir-201410, NCT02097186)
- Criterion (exclusion): Significant upper limb peripheral arterial disease
- Expert: **not excluded**, evidence []
- Model: **not excluded**, evidence [2, 3]
- Raw output: `{"evidence": [2, 3], "label": "not excluded"}`
- Gold evidence text: (none)
- Predicted evidence text: [2] Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered.; [3] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- Analysis: _write your interpretation here_

## Case 6: right_label_wrong_evidence  (annotation 179, sigir-201410, NCT02264964)
- Criterion (exclusion): Had been performed central venous puncture or catheterization before.
- Expert: **not excluded**, evidence [0]
- Model: **not excluded**, evidence []
- Raw output: `{"evidence": [], "label": "not excluded"}`
- Gold evidence text: [0] A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.
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

## Case 10: contradiction_same_evidence_wrong_label  (annotation 187, sigir-201411, NCT01526382)
- Criterion (inclusion): A respiratory rate of at least 20/min or a PaCO2 of 32mmHg or lower or the use of mechanical ventilation;
- Expert: **included**, evidence [2]
- Model: **not enough information**, evidence [2]
- Raw output: `{"evidence": [2], "label": "not enough information"}`
- Gold evidence text: [2] On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic.
- Predicted evidence text: [2] On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic.
- Analysis: _write your interpretation here_

## Case 11: contradiction_same_evidence_wrong_label  (annotation 762, sigir-201513, NCT00315042)
- Criterion (inclusion): A sore and scratchy throat and/or pain on swallowing (odynophagia) together with at least 2 of the following clinical signs:
- Expert: **not enough information**, evidence [0]
- Model: **included**, evidence [0]
- Raw output: `{"evidence": [0], "label": "included"}`
- Gold evidence text: [0] A 5-year-old boy presents to the emergency department with complaints of progressively worsening dysphagia, drooling, fever and vocal changes.
- Predicted evidence text: [0] A 5-year-old boy presents to the emergency department with complaints of progressively worsening dysphagia, drooling, fever and vocal changes.
- Analysis: _write your interpretation here_

## Case 12: polarity_error  (annotation 164, sigir-201410, NCT02097186)
- Criterion (inclusion): Patients undergoing surgical lower limb revascularisation (suprainguinal or infrainguinal)
- Expert: **not included**, evidence []
- Model: **included**, evidence [0, 2]
- Raw output: `{"evidence": [0, 2], "label": "included"}`
- Gold evidence text: (none)
- Predicted evidence text: [0] A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.; [2] Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered.
- Analysis: _write your interpretation here_
