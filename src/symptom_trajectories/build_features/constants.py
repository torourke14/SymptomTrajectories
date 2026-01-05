import re

DEFAULT_PDC_WINDOW_DAYS = 90
DEFAULT_MAX_SEQUENCE_DAYS = 128

PHQ9_TOTAL_LOINC = "44261-6"  # PHQ-9 total score [Reported]
PHQ9_ITEM_CODES = {
  "44250-9", "44255-8", "44259-0", "44254-1", "44251-7",
  "44258-2", "44252-5", "44253-3", "44260-8",
  "44261-6",
}
OBS_LOINC_CODES = {
  "phq9":   ["44261-6"],
  "phq2":   ["55758-7"],
  "gad7":   ["70274-6"],
  "auditc": ["75626-2"],
  "dast10": ["82667-7"],
  "smoking": ["72166-2"]
}
SCREENING_KWS = {
  "phq9":   ["phq9", "phq-9", "phq9 total", "phq-9 total"],
  "phq2":   ["phq2", "phq2 total", "phq-2 total"],
  "gad7":   ["gad7", "gad-7", "gad7 total", "gad7-total"],
  "auditc": ["auditc", "audit-c", "auditc total", "audit-c total"],
  "dast10": ["dast10", "dast-10", "dast10 total", "dast-10 total"],
}
SMOKING_STATUS_CODE = "72166-2"
SMOKER_COLLAPSE = {
  "never smoker": "never", "never": "never",
  "former smoker": "former","past": "former",
  "current some day smoker": "current", "current every day smoker": "current",
  "current": "current"
}
ALCOHOL_COLLAPSE = {
  "none": "none","no use": "none",
  "moderate": "moderate","light": "moderate",
  "heavy": "heavy","risky": "heavy","hazardous": "heavy"
}
PREGNANCY_CODES = { "2106-3", "80384-1", "2112-1" }
ENCOUNTER_TYPES = {
  "emergency": "EMERGENCY",
  "inpatient": "INPATIENT",
  "ambulatory": "OUTPATIENT",
  "outpatient": "OUTPATIENT",
  "urgentcare": "URGENTCARE",
  "wellness": "WELLNESS",
}
ANTIDEPRESSANT_KEYWORDS = [
  "alprazolam", "xanax", "amitriptyline", "elavil", "aripiprazole",
  "abilify", "asenapine", "saphris", "atomoxetine", "strattera",
  "brexpiprazole", "rexulti", "bupropion", "wellbutrin", "buspirone",
  "buspar", "buprenorphine", "sublocade", "citalopram", "celexa",
  "clonazepam", "klonopin", "desvenlafaxine", "pristiq", "deutetrabenazine",
  "austedo", "diazepam", "valium", "duloxetine", "cymbalta",
  "escitalopram", "lexapro", "esketamine", "spravato", "fluoxetine",
  "prozac", "gabapentin", "neurontin", "haloperidol", "haldol",
  "hydroxyzine", "vistaril", "lamotrigine", "lamictal", "levomilnacipran",
  "lofexidine", "lucemyra", "lorazepam", "ativan", "mirtazapine",
  "remeron", "milnacipran", "naltrexone", "vivitrol", "nortriptyline",
  "olanzapine", "zyprexa", "paroxetine", "paxil", "pregabalin",
  "lyrica", "quetiapine", "seroquel", "risperidone", "risperdal",
  "sertraline", "zoloft", "trazodone", "desyrel", "venlafaxine",
  "effexor", "vortioxetine", "vilazodone", "imipramine", "desipramine",
  "clomipramine", "doxepin"
]
ANTIDEPRESSANT_REGEX = re.compile(r"(" + "|".join([re.escape(k) for k in ANTIDEPRESSANT_KEYWORDS]) + r")", re.I)
