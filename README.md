## Requirements
Running this code requires access to MIMIC-IV data. To get access, you have to complete a certificate course in *Human Research* and *Data or Specimens Research* set by MIT Technology Affiliates. This must be completed and then requested through PhysioNet.

## Datasets
- MIMIC-IV: This repository utilizes BigQuery to read/write


## Generating Synthetic Data
See https://github.com/synthetichealth/synthea/wiki/Developer-Setup-and-Running
  1. git clone https://github.com/synthetichealth/synthea
  2. cd synthea
  3. Ensure (per README) Java SDK v11-v17 is installed
  4. Under "*data/synthetic/synthea/src/main/resources/synthea.properties*", change *exporter.csv.export* to true
1. on Windows, run .\gradlew run --args="-p 5000 -d ..\modules --exporter.csv.export=true"
  - -p 7500 -- sets population size
  - -d ../modules -- path to folder to add synthetic data as part of creation. For this project, we add synthetic PHQ-9 data from "/modules/phq9/phq9.json" module (included in reference docs)
  - Files are output under *output/csv* in the repo.
2. Using 7-Zip, compress (.gz) each of **patients, encounters, conditions, medications, and observations**individually. Upload to path of your liking (This code runs with uploaded to *submission/synthetic-data*)