# Error analysis (base_test)

Category counts:

- **polarity_error**: 67
- **decision_without_information**: 59
- **right_label_wrong_evidence**: 41
- **applicability_error**: 21
- **missed_information**: 4
- **contradiction_same_evidence_wrong_label**: 2

## Case 1: polarity_error  (annotation 161, sigir-201410, NCT02097186)
- Criterion (inclusion): Patients undergoing elective carotid endarterectomy
- Expert: **not included**, evidence []
- Model: **included**, evidence [1, 2, 3]
- Raw output: `{"evidence": [1, 2, 3], "label": "included"}`
- Gold evidence text: (none)
- Predicted evidence text: [1] She is now complaining of a cool right foot.; [2] Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered.; [3] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- Analysis: _write your interpretation here_

## Case 2: polarity_error  (annotation 162, sigir-201410, NCT02097186)
- Criterion (inclusion): Patients undergoing open abdominal aortic aneurysm repair
- Expert: **not included**, evidence []
- Model: **included**, evidence [1, 2, 3]
- Raw output: `{"evidence": [1, 2, 3], "label": "included"}`
- Gold evidence text: (none)
- Predicted evidence text: [1] She is now complaining of a cool right foot.; [2] Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered.; [3] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- Analysis: _write your interpretation here_

## Case 3: decision_without_information  (annotation 169, sigir-201410, NCT02097186)
- Criterion (exclusion): Patients with an estimated pre-operative glomerular filtration rate < 30mls/min/1.73m2
- Expert: **not enough information**, evidence []
- Model: **not excluded**, evidence [1, 2, 3]
- Raw output: `{"evidence": [1, 2, 3], "label": "not excluded"}`
- Gold evidence text: (none)
- Predicted evidence text: [1] She is now complaining of a cool right foot.; [2] Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered.; [3] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- Analysis: _write your interpretation here_

## Case 4: decision_without_information  (annotation 174, sigir-201410, NCT02097186)
- Criterion (exclusion): Patients with previous axillary surgery
- Expert: **not enough information**, evidence []
- Model: **excluded**, evidence [1, 2]
- Raw output: `{"evidence": [1, 2], "label": "excluded"}`
- Gold evidence text: (none)
- Predicted evidence text: [1] She is now complaining of a cool right foot.; [2] Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered.
- Analysis: _write your interpretation here_

## Case 5: right_label_wrong_evidence  (annotation 159, sigir-201410, NCT02097186)
- Criterion (inclusion): Age greater than 18 years
- Expert: **included**, evidence [0]
- Model: **included**, evidence [1, 2]
- Raw output: `{"evidence": [1, 2], "label": "included"}`
- Gold evidence text: [0] A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.
- Predicted evidence text: [1] She is now complaining of a cool right foot.; [2] Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered.
- Analysis: _write your interpretation here_

## Case 6: right_label_wrong_evidence  (annotation 186, sigir-201411, NCT01526382)
- Criterion (inclusion): Heart rate of at least 90/min;
- Expert: **included**, evidence [2]
- Model: **included**, evidence [5]
- Raw output: `{"evidence": [5], "label": "included"}`
- Gold evidence text: [2] On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic.
- Predicted evidence text: [5] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- Analysis: _write your interpretation here_

## Case 7: applicability_error  (annotation 165, sigir-201410, NCT02097186)
- Criterion (exclusion): Pregnancy
- Expert: **not applicable**, evidence []
- Model: **excluded**, evidence [1, 2]
- Raw output: `{"evidence": [1, 2], "label": "excluded"}`
- Gold evidence text: (none)
- Predicted evidence text: [1] She is now complaining of a cool right foot.; [2] Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered.
- Analysis: _write your interpretation here_

## Case 8: applicability_error  (annotation 175, sigir-201410, NCT02264964)
- Criterion (inclusion): Chronic renal failure requiring hemodialysis.
- Expert: **not enough information**, evidence []
- Model: **not applicable**, evidence [1, 2, 3]
- Raw output: `{"evidence": [1, 2, 3], "label": "not applicable"}`
- Gold evidence text: (none)
- Predicted evidence text: [1] She is now complaining of a cool right foot.; [2] Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered.; [3] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- Analysis: _write your interpretation here_

## Case 9: missed_information  (annotation 317, sigir-201417, NCT01745731)
- Criterion (exclusion): Proliferative retinopathy.
- Expert: **not excluded**, evidence []
- Model: **not enough information**, evidence []
- Raw output: `{"evidence": [], "label": "not enough information"}`
- Gold evidence text: (none)
- Predicted evidence text: (none)
- Analysis: _write your interpretation here_

## Case 10: missed_information  (annotation 686, sigir-20158, NCT02562040)
- Criterion (exclusion): Failure to thrive, defined as either height or weight being below the 5th percentile for age and gender.
- Expert: **not excluded**, evidence []
- Model: **not enough information**, evidence []
- Raw output: `{"evidence": [], "label": "not enough information"}`
- Gold evidence text: (none)
- Predicted evidence text: (none)
- Analysis: _write your interpretation here_

## Case 11: contradiction_same_evidence_wrong_label  (annotation 326, sigir-201417, NCT02556359)
- Criterion (exclusion): Refusal to consent.
- Expert: **not excluded**, evidence [5]
- Model: **excluded**, evidence [5]
- Raw output: `{"evidence": [5], "label": "excluded"}`
- Gold evidence text: [5] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- Predicted evidence text: [5] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- Analysis: _write your interpretation here_

## Case 12: contradiction_same_evidence_wrong_label  (annotation 831, sigir-201516, NCT00839124)
- Criterion (inclusion): Oxygen saturation of > 94 % and normal blood pressure (Systolic between 150 - 90, Diastolic between 90-60 mm Hg)
- Expert: **not enough information**, evidence [7]
- Model: **included**, evidence [7]
- Raw output: `{"evidence": [7], "label": "included"}`
- Gold evidence text: [7] O2 sats are 100% on room air.
- Predicted evidence text: [7] O2 sats are 100% on room air.
- Analysis: _write your interpretation here_
