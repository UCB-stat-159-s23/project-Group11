[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/LiaEl886)

# Reproducible Research Project for Institution Retention Rates

Binder link: [![DOI](https://mybinder.org/v2/gh/UCB-stat-159-s23/project-Group11.git/HEAD)

Github page:

Dataset link: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7857257.svg)](https://doi.org/10.5281/zenodo.7857257)

This is a project attempting to analyze the factors that affect student retention at universities in a reproducible manner. We conduct exploratory data analysis, compute feature importance, and implement logistic regression for prediction. The data originates from the U.S. Department of Education College Scoreboard through the Institution-level data files for 1996-97 through 2020-21 containing aggregate data for each institution. The dataset includes information on institutional characteristics, enrollment, student aid, costs, and student outcomes. You can find more information here: https://collegescorecard.ed.gov/data/

## Repository Structure
* `/data`: cleaned data csv files
* `/figures`: all generated figures as png files
* Jupyter Notebooks:
    - `main.ipynb`: the main narrative notebook of the research project
    - `EDA.ipynb`: code for exploratory data analysis and for EDA related figures
    - `Variable_Analysis.ipynb`: code for logistic regression model and their corresponding figures
* Python utility package tools:
    - `/tools`: code and tests for python package
    - Setup files: `setup.py`, `setup.cfg`, `pyproj.toml`
* Environment files: `environment.yml`, `envsetup.sh`
* Jupyter Book: `_config.yml`, `_toc.yml`, `conf.py`, `postBuild`, `requirements.txt`
* `contribution_statement.md`: authors' contributions

## Dataset Attribute Information
