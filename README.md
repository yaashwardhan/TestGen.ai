# TestGen.ai :scroll::pencil2:

<a href="https://huggingface.co/docs/transformers/index"><img src="https://img.shields.io/badge/Powered%20by-Transformers-orange.svg"/></a> [![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/) ![GitHub last commit (branch)](https://img.shields.io/github/last-commit/yaashwardhan/BrainStain.ai/main?color=blue)

A host can input a paragraph of words and and the platform then generates different types of quizzes from it. For this, keywords were extracted from the paragraph using unsupervised keyphrase extraction with multipartite graphs, which were then passed to a system that employs a transformer model that is finetuned using transfer learning on the SQuAD dataset, using the T5 model and tokenizer to generate questions pertaining to the extracted keyword. Using sense2vec, Normalized Levenshtein distance algorithm and Maximal Marginal Relevance algorithm (cosine similarity), dissimilar distractors were generated to create incorrect options for the question. Using BERT overcame word sense disambiguation for distractor sense classification. Flask was used as the Python app to generate the tests, while Ajax was employed as a handler between the website and the question generation models. JavaScript was used for client-side scripting.

## Dependencies

| Package              | Tested version |
|----------------------|----------------|
| torch                | 1.13.1         |
| transformers         | 4.20.1         |
| numpy                | 1.23.3         |
| requests             | 2.28.1         |
| sense2vec            | 2.0.1          |
| similarity           | 0.0.1          |
| spacy                | 3.3.2          |
| strsim               | 0.0.3          |
| nltk                 | 3.8.1          |
| pandas               | 1.5.0          |
| pke                  | 2.0.0          |
| flashtext            | 2.7            |
| pytorch_lightning    | 1.2.10         |


## Installation

```bash
  $ git clone https://github.com/yaashwardhan/TestGen.ai.git
```
Navigate to the project directory.
```bash
  $ cd TestGen.ai/
```
Create a new Conda environment.
```bash
  $ conda create --name environment-name python=3.10.10
```
Note: Replace environment-name with the name you want to give your Conda environment.

Activate the newly created environment.
```bash
  $ conda activate environment-name
```
Install the project dependencies from the requirements.txt file.
```bash
  $ pip install -r requirements.txt
```
Verify that all the required packages have been installed correctly by running:
```bash
  $ pip freeze
```
