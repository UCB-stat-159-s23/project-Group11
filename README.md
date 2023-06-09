[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/LiaEl886)

# Reproducible Research Project for Institution Retention Rates

Binder link: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/UCB-stat-159-s23/project-Group11.git/HEAD?labpath=main.ipynb)

Github page: [https://ucb-stat-159-s23.github.io/project-Group11/](https://ucb-stat-159-s23.github.io/project-Group11/main.html)

Dataset link: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7857257.svg)](https://doi.org/10.5281/zenodo.7857257)

## Project Goal

This is a project attempting to analyze the factors that affect student retention at universities in a reproducible manner. We conduct exploratory data analysis, compute feature importance, and implement logistic regression for prediction. The data originates from the U.S. Department of Education College Scoreboard through the Institution-level data files for 1996-97 through 2020-21 containing aggregate data for each institution. The dataset includes information on institutional characteristics, enrollment, student aid, costs, and student outcomes. You can find more information here: https://collegescorecard.ed.gov/data/

## Useful Commands
* To run all tests for utility functions, run ``` pytest tools ``` from the terminal in the root directory.
* To create the environment, run ``` make env ``` from the terminal in the root directory.
* In addition, you may run ``` make html ``` to build the jupyter-book html, ``` make clean ``` to remove figures and html files, and ``` make all ``` to run all the aforementioned ```make``` commands.

## Repository Structure
* `/data`: cleaned data csv files
* `/figures`: all generated figures as png files
* Jupyter Notebooks:
    - `main.ipynb`: the main narrative notebook of the research project
    - `EDA.ipynb`: introduction, data description and some basic exploratory data analysis of retention rates
    - `EDA_control_of_school`: the code of the control of the school for exploratory data analysis and EDA related figures
    - `EDA_in_out_state_tuition`: the code of in/out state tuition & fee for exploratory data analysis and for EDA related figures
    - `EDA_rece_loans`: the code of individuals receiving federal loans for exploratory data analysis and for EDA related figures of percentage
    - `EDA_parent_edu`: the code of parent education for exploratory data analysis and for EDA related figures
    - `EDA_program_offered`: the code of programs offered for exploratory data analysis and for EDA related figures
    - `EDA_num_var`: feature analysis on numberical variables
    - `Variable_Analysis_1.ipynb`: code for logistic regression model and their corresponding figures of four-year institutions retention rates
    - `Variable_Analysis_2.ipynb`: code for logistic regression model and their corresponding figures of less than four-year institutions retention rates
* Python utility package tools:
    - `/tools`: code and tests for python package
    - Setup files: `setup.py`, `setup.cfg`, `pyproj.toml`
* Environment files: `environment.yml`, `envsetup.sh`
* Jupyter Book: `_config.yml`, `_toc.yml`, `conf.py`, `postBuild`, `requirements.txt`
* `contribution_statement.md`: authors' contributions

## Dataset Attribute Information
* `UNITID`: Unit ID for institution
* `CONTROL`: Control of institution
* `CCUGPROF`: Carnegie Classification -- undergraduate profile
* `CCSIZSET`: Carnegie Classification -- size and setting
* `ADM_RATE`: Admission Rate
* `SAT_AVG`: Average SAT equivalent score of students admitted
* `UG`: Enrollment of all undergraduate students
* `UGDS_[...]`: Total share of enrollment of undergraduate degree-seeking students who are [...]
    - UGDS_WHITE (white), UGDS_BLACK (black), UGDS_HISP (Hispanic), UGDS_ASIAN (Asian), UGDS_AIAN (American Indian/Alaska Native), UGDS_NHPI (Native Hawaiian/Pacific Islander), UGDS_2MOR (two or more races), UGDS_NRA (non-resident aliens), UGDS_UNKN (unknown), UGDS_WHITENH (white non-Hispanic)
* `NPT4_PUB`: Average net price for Title IV institutions (public institutions)
* `NPT4_PRIV`: Average net price for Title IV institutions (private for-profit and nonprofit institutions)
* `TUITIONFEE_IN`: In-state tuition and fees
* `TUITIONFEE_OUT`: Out-of-state tuition and fees
* `AVGFACSAL`: Average faculty salary
* `PCTPELL`: Percentage of undergraduates who receive a Pell grant
* `RET_FT4`: First-time, full-time student retention rate at four-year institutions
* `RET_FTL4`: First-time, full-time student retention rate at less-than-four-year institutions
* `PCTFLOAN`: Percent of all undergraduate students receiving a federal student loan
* `PAR_ED_PCT_MS`: Percent of students whose parents' highest educational level is middle school
* `PAR_ED_PCT_HS`: Percent of students whose parents' highest educational level is high school
* `PAR_ED_PCT_PS`: Percent of students whose parents' highest educational level was is some form of postsecondary education
* `DEP_INC_AVG`: Average family income of dependent students in real 2015 dollars
* `IND_INC_AVG`: Average family income of independent students in real 2015 dollars
* `GRAD_DEBT_MDN`: The median debt for students who have completed
* `WDRAW_DEBT_MDN`: The median debt for students who have not completed
* `FAMINC`: Average family income
* `MD_FAMINC`: Median family income
* `PRGMOFR`: Number of programs offered

For more information on the additional variables, please refer to the [Data Dictionary](https://collegescorecard.ed.gov/data/). Make sure to download the `Most Recent Institution-Level Data` data dictionary and you can find all the descriptions of the variables in the `Institution_Data_Dictionary` tab.

## Testing 

1. To guarantee that the correct environment is in place, execute the `make env` command. If an outdated version of the environment is already set up, or if you need to modify the environment, activate a separate environment using `conda activate` and then execute `make remove-env` first to remove the prior setup.

2. After running conda activate `final_proj`, run `conda install -c anaconda pytest`.

3. Run `pytest tools`.

## Running Instructions
To run the main notebook, you have two options: you can either use the provided [binder link](https://mybinder.org/v2/gh/UCB-stat-159-s23/project-Group11.git/HEAD?labpath=main.ipynb), or you can [clone the repository](https://github.com/UCB-stat-159-s23/project-Group11.git) and run the notebook locally. If you choose to run the notebook locally, we have also included a makefile with instructions to help streamline the process.
 
## License
The project is released under the BSD 3-clause License.