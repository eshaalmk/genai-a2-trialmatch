# Error analysis (finetuned_val)

Category counts:

- **unsupported_decision**: 111
- **right_label_wrong_evidence**: 37
- **decision_without_information**: 32
- **applicability_error**: 12
- **missed_information**: 11
- **contradiction_same_evidence_wrong_label**: 9
- **polarity_error**: 3

## Case 1: unsupported_decision  (annotation 6, sigir-20141, NCT01397994)
- Criterion (exclusion): Valvular heart disease and cardiomyopathy
- Expert: **not excluded**, evidence []
- Model: **not excluded**, evidence []
- Raw output: `{"evidence": [], "label": "not excluded"}`
- Gold evidence text: (none)
- Predicted evidence text: (none)
- Analysis: _write your interpretation here_

## Case 2: unsupported_decision  (annotation 7, sigir-20141, NCT01397994)
- Criterion (exclusion): Myocardial infarction in < 6 months
- Expert: **not excluded**, evidence []
- Model: **not excluded**, evidence []
- Raw output: `{"evidence": [], "label": "not excluded"}`
- Gold evidence text: (none)
- Predicted evidence text: (none)
- Analysis: _write your interpretation here_

## Case 3: right_label_wrong_evidence  (annotation 0, sigir-20141, NCT01397994)
- Criterion (inclusion): Patients of chronic stable angina with abnormal Exercise Myocardial Perfusion Spect Scan with reversible and partially reversible ischemic changes.
- Expert: **not enough information**, evidence [0, 1, 2]
- Model: **not enough information**, evidence []
- Raw output: `{"evidence": [], "label": "not enough information"}`
- Gold evidence text: [0] A 58-year-old African-American woman presents to the ER with episodic pressing/burning anterior chest pain that began two days earlier for the first time in her life.; [1] The pain started while she was walking, radiates to the back, and is accompanied by nausea, diaphoresis and mild dyspnea, but is not increased on inspiration.; [2] The latest episode of pain ended half an hour prior to her arrival.
- Predicted evidence text: (none)
- Analysis: _write your interpretation here_

## Case 4: right_label_wrong_evidence  (annotation 5, sigir-20141, NCT01397994)
- Criterion (exclusion): Hypertension of > 170/100 mm of Hg
- Expert: **not excluded**, evidence []
- Model: **not excluded**, evidence [3]
- Raw output: `{"evidence": [3], "label": "not excluded"}`
- Gold evidence text: (none)
- Predicted evidence text: [3] She is known to have hypertension and obesity.
- Analysis: _write your interpretation here_

## Case 5: decision_without_information  (annotation 8, sigir-20141, NCT01397994)
- Criterion (exclusion): Unstable angina
- Expert: **not enough information**, evidence [0, 1, 2]
- Model: **not excluded**, evidence [0, 1, 2]
- Raw output: `{"evidence": [0, 1, 2], "label": "not excluded"}`
- Gold evidence text: [0] A 58-year-old African-American woman presents to the ER with episodic pressing/burning anterior chest pain that began two days earlier for the first time in her life.; [1] The pain started while she was walking, radiates to the back, and is accompanied by nausea, diaphoresis and mild dyspnea, but is not increased on inspiration.; [2] The latest episode of pain ended half an hour prior to her arrival.
- Predicted evidence text: [0] A 58-year-old African-American woman presents to the ER with episodic pressing/burning anterior chest pain that began two days earlier for the first time in her life.; [1] The pain started while she was walking, radiates to the back, and is accompanied by nausea, diaphoresis and mild dyspnea, but is not increased on inspiration.; [2] The latest episode of pain ended half an hour prior to her arrival.
- Analysis: _write your interpretation here_

## Case 6: decision_without_information  (annotation 15, sigir-20141, NCT01397994)
- Criterion (exclusion): Pregnant and nursing women
- Expert: **not enough information**, evidence []
- Model: **not excluded**, evidence []
- Raw output: `{"evidence": [], "label": "not excluded"}`
- Gold evidence text: (none)
- Predicted evidence text: (none)
- Analysis: _write your interpretation here_

## Case 7: applicability_error  (annotation 24, sigir-20141, NCT00149227)
- Criterion (exclusion): Patients with IHD within 6 months after percutaneous coronary intervention(PCI), and who are stable but are going to implement PCI or coronary artery bypass grafting(CABG)
- Expert: **not applicable**, evidence []
- Model: **not excluded**, evidence []
- Raw output: `{"evidence": [], "label": "not excluded"}`
- Gold evidence text: (none)
- Predicted evidence text: (none)
- Analysis: _write your interpretation here_

## Case 8: applicability_error  (annotation 33, sigir-20142, NCT02618655)
- Criterion (inclusion): full of physical examination and laboratory examination have been carried out after one week，but still cannot make a definite diagnosis
- Expert: **not applicable**, evidence []
- Model: **not enough information**, evidence [4, 5]
- Raw output: `{"evidence": [4, 5], "label": "not enough information"}`
- Gold evidence text: (none)
- Predicted evidence text: [4] On examination he is in respiratory distress and has bronchial respiratory sounds on the left.; [5] A chest x-ray shows bilateral lung infiltrates.
- Analysis: _write your interpretation here_

## Case 9: missed_information  (annotation 21, sigir-20141, NCT00149227)
- Criterion (inclusion): Clinical diagnosis of hypertension
- Expert: **included**, evidence [3]
- Model: **not enough information**, evidence [3]
- Raw output: `{"evidence": [3], "label": "not enough information"}`
- Gold evidence text: [3] She is known to have hypertension and obesity.
- Predicted evidence text: [3] She is known to have hypertension and obesity.
- Analysis: _write your interpretation here_

## Case 10: missed_information  (annotation 22, sigir-20141, NCT00149227)
- Criterion (inclusion): Clinical diagnosis of one or more risk factors, such as diabetes, smoking habit, lipid metabolism abnormality, history of ischemic heart disease (IHD) or cerebrovascular disease, obesity (BMI>25), chronic heart failure (NYHA II-III), and electrocardiogram (ECG) abnormality (LVH)
- Expert: **included**, evidence [3, 4]
- Model: **not enough information**, evidence [0, 3]
- Raw output: `{"evidence": [0, 3], "label": "not enough information"}`
- Gold evidence text: [3] She is known to have hypertension and obesity.; [4] She denies smoking, diabetes, hypercholesterolemia, or a family history of heart disease.
- Predicted evidence text: [0] A 58-year-old African-American woman presents to the ER with episodic pressing/burning anterior chest pain that began two days earlier for the first time in her life.; [3] She is known to have hypertension and obesity.
- Analysis: _write your interpretation here_

## Case 11: polarity_error  (annotation 41, sigir-20142, NCT00711399)
- Criterion (exclusion): Respiratory distress
- Expert: **excluded**, evidence [4]
- Model: **not excluded**, evidence [4]
- Raw output: `{"evidence": [4], "label": "not excluded"}`
- Gold evidence text: [4] On examination he is in respiratory distress and has bronchial respiratory sounds on the left.
- Predicted evidence text: [4] On examination he is in respiratory distress and has bronchial respiratory sounds on the left.
- Analysis: _write your interpretation here_

## Case 12: polarity_error  (annotation 263, sigir-201415, NCT00180739)
- Criterion (exclusion): 13. Evidence of uterine pathology other than leiomyoma
- Expert: **excluded**, evidence [5]
- Model: **not excluded**, evidence [4]
- Raw output: `{"evidence": [4], "label": "not excluded"}`
- Gold evidence text: [5] The differential diagnosis includes vesicular mole vs fibroid degeneration.
- Predicted evidence text: [4] The ultrasound reports enlarged uterus (12 cm x 9 cm x 7 cms) with multiple cystic areas in the interior.
- Analysis: _write your interpretation here_
