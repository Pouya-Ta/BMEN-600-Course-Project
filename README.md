# BMEN-600-Course-Project
This repo contains the course materials and/or information and codes of our team project. 

Collaborators (without order):
## Team Name 
Seventieth

## Team Members Present
1- Luke Fatovich
2- Pouya Taghipour L. 
3- Hamid Salmani
4- Dera Okemeziem
5- David A. Martinez G.

## Candidate Project 1

### Biomedical Problem 

Alzheimer’s disease (AD) and frontotemporal dementia (FTD) are clinically similar in the early stages, and distinguishing between them currently requires neuropsychological evaluation, neuroimaging, or cerebrospinal fluid (CSF) biomarkers, all of which are expensive or invasive. Resting EEG is inexpensive, noninvasive, and available in small clinics, so an EEG-based marker that can distinguish between AD, FTD, and healthy controls (HC) would be useful as a triage tool or to aid in differential diagnosis.

### Possible Research Question

Do foundational EEG models, using only a single linear probe, outperform classic, hand-designed features in the differential diagnosis of AD/FTD/HC when data are limited?

### Dataset 

OpenNeuro ds004504: Resting-state EEG (eyes closed) with 19 channels (10-20 system), from approximately 88 participants: about 36 with AD, 23 with FTD, and 29 controls. It includes raw data and a version that has already been filtered and cleaned using ICA. The metadata includes age, sex, and MMSE score for each subject.

### Biggest Uncertainty

The greatest risk is that the result will be more optimistic than reality due to the size and structure of the data

  - Cross-epoch information leakage: With only ~88 subjects, if a subject’s training and test epochs overlap, the accuracy is inflated. Validation by subject is needed (leave-subject-out or nested CV), and even then, the variance will be high.
  - No external validation: all data comes from a single site and a single team, so I don’t know if the findings generalize to other clinics or setups.

## Candidate Project 2

### Biomedical Problem 

### Possible Research Question

### Dataset 

### Biggest Uncertainty
