# All settings and constants for the project

# Model settings
RANDOM_STATE = 42
N_SPLITS     = 5      # Number of folds in cross-validation
N_CLASSES    = 3      # HIGH / INTERMEDIARY / LOW

# MCC threshold
MCC_THRESHOLD = 0.5   # Selected threshold τ

# Target
TARGET_COLUMN = "CVD Risk Level"
TARGET_CLASS  = "HIGH"

# Columns to drop (collinear or derived features)
COLUMNS_TO_DROP = [
    "Blood Pressure (mmHg)",
    "CVD Risk Score",
    "BMI",
    "Height (cm)",
]

# Zone classification
ZONE0_PROBA_GAP = 0.05  # minimum probability gap between 2nd and 3rd class for Zone 0

# Cost parameters Bangladesh (USD)
# Sources: Sarker et al. 2025; Husain et al. 2022; Hasan et al. 2025
BD_EVENT_COST      = 1053   # per CVD event (out-of-pocket)
BD_PREVENTION_COST = 37     # per patient/year (preventive treatment)
BD_SCREENING_COST  = 5      # per patient
BD_GP_COST         = 3.40   # per GP consultation
BD_CURRENCY        = "USD"

# Cost parameters Germany (EUR)
# Sources: Schmid et al. 2015; Dressel et al. 2024; Schuetz et al. 2013
DE_EVENT_COST      = 7800   # per CVD event (acute hospital)
DE_PREVENTION_COST = 100    # per patient/year (preventive treatment)
DE_SCREENING_COST  = 43     # per patient
DE_GP_COST         = 35     # per GP consultation
DE_CURRENCY        = "EUR"