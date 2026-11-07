# Error analysis (base_test)

Category counts:

- **missed_information**: 88
- **right_label_wrong_evidence**: 27
- **applicability_error**: 15
- **polarity_error**: 13
- **decision_without_information**: 11

## Case 1: missed_information  (annotation 167, sigir-201410, NCT02097186)
- Criterion (exclusion): Previous history of upper limb deep vein thrombosis
- Expert: **not excluded**, evidence []
- Model: **not enough information**, evidence []
- Raw output: `{"evidence": [], "label": "not enough information"}`
- Gold evidence text: (none)
- Predicted evidence text: (none)
- Analysis: _write your interpretation here_

## Case 2: missed_information  (annotation 170, sigir-201410, NCT02097186)
- Criterion (exclusion): Patients with a known history of myocarditis, pericarditis or amyloidosis
- Expert: **not excluded**, evidence []
- Model: **not enough information**, evidence []
- Raw output: `{"evidence": [], "label": "not enough information"}`
- Gold evidence text: (none)
- Predicted evidence text: (none)
- Analysis: _write your interpretation here_

## Case 3: right_label_wrong_evidence  (annotation 159, sigir-201410, NCT02097186)
- Criterion (inclusion): Age greater than 18 years
- Expert: **included**, evidence [0]
- Model: **included**, evidence [1, 2]
- Raw output: `{"evidence": [1, 2], "label": "included"}`
- Gold evidence text: [0] A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.
- Predicted evidence text: [1] She is now complaining of a cool right foot.; [2] Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered.
- Analysis: _write your interpretation here_

## Case 4: right_label_wrong_evidence  (annotation 186, sigir-201411, NCT01526382)
- Criterion (inclusion): Heart rate of at least 90/min;
- Expert: **included**, evidence [2]
- Model: **included**, evidence [5]
- Raw output: `{"evidence": [5], "label": "included"}`
- Gold evidence text: [2] On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic.
- Predicted evidence text: [5] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- Analysis: _write your interpretation here_

## Case 5: applicability_error  (annotation 165, sigir-201410, NCT02097186)
- Criterion (exclusion): Pregnancy
- Expert: **not applicable**, evidence []
- Model: **not enough information**, evidence []
- Raw output: `{"evidence": [], "label": "not enough information"}`
- Gold evidence text: (none)
- Predicted evidence text: (none)
- Analysis: _write your interpretation here_

## Case 6: applicability_error  (annotation 422, sigir-201422, NCT00236912)
- Criterion (inclusion): If female, using birth control
- Expert: **not applicable**, evidence [3]
- Model: **not enough information**, evidence []
- Raw output: `{"evidence": [], "label": "not enough information"}`
- Gold evidence text: [3] She has had no sexual partners and her menses are regular.
- Predicted evidence text: (none)
- Analysis: _write your interpretation here_

## Case 7: polarity_error  (annotation 161, sigir-201410, NCT02097186)
- Criterion (inclusion): Patients undergoing elective carotid endarterectomy
- Expert: **not included**, evidence []
- Model: **included**, evidence [1, 2, 3]
- Raw output: `{"evidence": [1, 2, 3], "label": "included"}`
- Gold evidence text: (none)
- Predicted evidence text: [1] She is now complaining of a cool right foot.; [2] Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered.; [3] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- Analysis: _write your interpretation here_

## Case 8: polarity_error  (annotation 162, sigir-201410, NCT02097186)
- Criterion (inclusion): Patients undergoing open abdominal aortic aneurysm repair
- Expert: **not included**, evidence []
- Model: **included**, evidence [1, 2, 3]
- Raw output: `{"evidence": [1, 2, 3], "label": "included"}`
- Gold evidence text: (none)
- Predicted evidence text: [1] She is now complaining of a cool right foot.; [2] Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered.; [3] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- Analysis: _write your interpretation here_

## Case 9: decision_without_information  (annotation 189, sigir-201411, NCT01526382)
- Criterion (inclusion): at least 1 of 3 signs of hypoperfusion (urine output < 0.5mL/kg of body weight per hour for 1 hour or more; neurologic dysfunction defined by confusion, psychosis, or a Glasgow coma scale score of ≤ 6; plasma lactate higher than the upper limit of the normal value).
- Expert: **not enough information**, evidence []
- Model: **included**, evidence [1, 2, 3, 4, 5]
- Raw output: `{"evidence": [1, 2, 3, 4, 5], "label": "included"}`
- Gold evidence text: (none)
- Predicted evidence text: [1] She denies trauma.; [2] On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic.; [3] Her body temperature is normal and her blood pressure is 80/60.; [4] Her right arm has no discoloration or movement limitation.; [5] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- Analysis: _write your interpretation here_

## Case 10: decision_without_information  (annotation 305, sigir-201417, NCT01745731)
- Criterion (inclusion): Standard analytical parameters, defined by: Leukocytes ≥ 3000 Neutrophils ≥ 1500 Platelets ≥ 100,000 Aspartate aminotransferase (AST) / Alanine aminotransferase (ALT) ≤ 1.5 standard range institution creatinine ≤ 1.5 mg / dl
- Expert: **not enough information**, evidence []
- Model: **included**, evidence [1, 2, 3, 4, 5]
- Raw output: `{"evidence": [1, 2, 3, 4, 5], "label": "included"}`
- Gold evidence text: (none)
- Predicted evidence text: [1] The physical examination is remarkable for tenderness and positive Murphy sign.; [2] Abdominal ultrasound shows hepatomegaly and abundant free intraperitoneal fluid.; [3] Exploratory laparotomy reveals a ruptured liver abscess, which is then surgically drained.; [4] After surgery, the patient is taken to the ICU.; [5] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- Analysis: _write your interpretation here_
