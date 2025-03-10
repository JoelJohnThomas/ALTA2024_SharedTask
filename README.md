# ALTA 2024 Shared Task

This project is about the 2024 Shared task from the Australasian Language Technology Association. The goal of this shared task is to develop automatic detection systems capable of identifying AI-generated sentences within hybrid articles containing both human-written and AI-generated content. Participants are challenged to create models that can accurately distinguish between human-authored and GPT-3.5-turbo-generated sentences in collaborative writing scenarios

## Dataset
The training data was primarily obtained from the publicly available dataset curated by Zeng et al (https://arxiv.org/abs/2307.12267) and was expanded to include some more data from CC-News dataset. The test set was privately collected by the organisers specifically from the news domain. 

Source : https://codalab.lisn.upsaclay.fr/competitions/19633#participate-get_starting_kit
Format :  A Json file containing id of the article, domain of the article, sentences in the articles and their correspoding labels(human/machine)
Features : Article ID (a number) , Domain("news"/"academic"), sentence(a string) , label("human"/"machine")
Size: Train set - 39 MB (212794 rows), Phase 1 test - 926 KB (4264 rows), Final test - 1327 KB (8652 rows)

## Prerequisites
- Python 3.8+
- Jupyter Notebook 
- Libraries: `spacy`, `matplotlib`, `pandas`, `os`, `re`, `glob`, `faststylometry`, `numpy`, `scikit-learn`, `texstat`, `json`, `math`, `seaborn`



