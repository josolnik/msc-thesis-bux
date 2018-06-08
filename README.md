### Automating interpretable feature engineering for predicting human behavior
##### Master thesis submitted in partial fulfillment for the degree MSc Information Studies, Data Science track @ University of Amsterdam

----
##### Research question: To what extent can we automate a generalizable and interpretable system for predicting human behavior?

----
### Re-use of the pipeline
1. API scripts combine the *utils.py* (generalizable script) and *bux\_utils\_filtered.py* (custom script), which is filtered for confidentiality.
2. *Training & Report - CLV prediction.ipynb* and *Scoring  - CLV prediction.ipynb* are built by using both API scripts as building blocks. They are used to build and test both pipelines.
3. *training.py* and *scoring.py* are both pipelines in the previous step, built as scripts in order to be run as jobs and produce scores either into csvs or in the database by using the established connnections
4. *window\_training\_scoring.py* is used to run both scripts from the last step on different time frames by using the window specified for both of the pipelines


More info available in the thesis document ( *automating\_interpretable_feature\_engineering.pdf*).