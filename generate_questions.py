import random
import string
import time
import json
import requests
import re
import itertools
import traceback
import textwrap
import xml.etree.ElementTree as et
from collections import OrderedDict

import numpy as np
import pandas as pd
import spacy
import torch
from similarity.normalized_levenshtein import NormalizedLevenshtein
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import nltk
from nltk import FreqDist
from nltk.corpus import brown
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet

nltk.download('brown')
nltk.download('popular')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

from typing import List, Tuple
from flashtext import KeywordProcessor
from sense2vec import Sense2Vec
import pke

from IPython.core.display import display, HTML


def generate_using_sense2vec(word, s2v):
    output = [] 
    # Preprocess the input word by removing punctuation and converting to lowercase
    word_preprocessed = word.translate(str.maketrans('', '', string.punctuation)).lower()
    # Generate a list of possible edits for the preprocessed word
    word_edits = edits(word_preprocessed)
    # Replace spaces in the word with underscores
    word = word.replace(' ', '_')
    # Use sense2vec to get the best sense for the word and find the most similar words
    sense = s2v.get_best_sense(word)
    most_similar = s2v.most_similar(sense, n=15)
    # Compare the most similar words with the preprocessed word and append them to the output list if they meet certain criteria
    compare_list = [word_preprocessed]
    for each_word in most_similar:
        append_word = each_word[0].split('|')[0].replace('_', ' ').strip()
        append_word_processed = append_word.lower().translate(str.maketrans('', '', string.punctuation))
        if append_word_processed not in compare_list and word_preprocessed not in append_word_processed and append_word_processed not in word_edits:
            output.append(append_word.title())
            compare_list.append(append_word_processed)
    # Remove duplicates from the output list while preserving the order of elements
    out = list(OrderedDict.fromkeys(output))
    return out



def get_distractors_as_options(answer: str, s2v) -> Tuple[List[str], str]:
    """
    Get distractors for a given answer using sense2vec.

    Parameters:
    answer (str): the answer word for which distractors need to be generated
    s2v: the sense2vec model

    Returns:
    distractors (List[str]): a list of distractors for the answer
    method (str): the method used to generate the distractors, either 'sense2vec' or 'None' if generation failed
    """
    distractors = []
    try:
        distractors = generate_using_sense2vec(answer, s2v)
        if len(distractors) > 0:
            print(f"Sense2vec_distractors successful for word: {answer}")
            return distractors, "sense2vec"
    except:
        print(f"Sense2vec_distractors failed for word: {answer}")
    return distractors, "None"


def sentence_tokenize(text: str) -> List[str]:
    """
    Tokenize a given text into sentences.

    Parameters:
    text (str): the text to tokenize

    Returns:
    sentences (List[str]): a list of sentences
    """
    sentences = [sent_tokenize(text)]
    sentences = [y for x in sentences for y in x]
    sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]
    return sentences

def generate_keyword_sentences(keywords: List[str], sentences: List[str]) -> dict:
    """
    Get sentences containing keywords from a list of sentences.

    Parameters:
    keywords (List[str]): a list of keywords
    sentences (List[str]): a list of sentences

    Returns:
    keyword_sentences (dict): a dictionary mapping each keyword to a list of sentences containing that keyword
    """
    keyword_processor = KeywordProcessor()
    keyword_sentences = {}
    for word in keywords:
        word = word.strip()
        keyword_sentences[word] = []
        keyword_processor.add_keyword(word)
    for sentence in sentences:
        keywords_found = keyword_processor.extract_keywords(sentence)
        for key in keywords_found:
            keyword_sentences[key].append(sentence)

    for key in keyword_sentences.keys():
        values = keyword_sentences[key]
        values = sorted(values, key=len, reverse=True)
        keyword_sentences[key] = values

    delete_keys = []
    for k in keyword_sentences.keys():
        if len(keyword_sentences[k]) == 0:
            delete_keys.append(k)
    for del_key in delete_keys:
        del keyword_sentences[del_key]

    return keyword_sentences


def word_determination(words_list, current_word, threshold, normalized_levenshtein):
    """
    Determine if a word is far enough from a list of words based on the normalized Levenshtein distance.

    Args:
        words_list (list): List of words to compare with.
        current_word (str): Word to compare.
        threshold (float): Threshold value to determine if the word is far enough.
        normalized_levenshtein (object): NormalizedLevenshtein object from the jellyfish library.

    Returns:
        bool: True if the word is far enough from all the words in the list, False otherwise.
    """
    score_list = [normalized_levenshtein.distance(word.lower(), current_word.lower()) for word in words_list]
    return min(score_list) >= threshold


def filter_similar(phrase_keys, max_phrases, normalized_levenshtein):
    """
    Filter a list of phrases based on their similarity using the normalized Levenshtein distance.

    Args:
        phrase_keys (list): List of phrases to filter.
        max_phrases (int): Maximum number of phrases to return.
        normalized_levenshtein (object): NormalizedLevenshtein object from the jellyfish library.

    Returns:
        list: List of filtered phrases.
    """
    filtered_phrases = [phrase_keys[0]]
    for phrase in phrase_keys[1:]:
        if word_determination(filtered_phrases, phrase, 0.8, normalized_levenshtein):
            filtered_phrases.append(phrase)
        if len(filtered_phrases) >= max_phrases:
            break
    return filtered_phrases

def get_nouns_multipartite(text):
    """
    Extract the top 10 candidate keyphrases from a given text using the MultipartiteRank algorithm.

    Args:
        text (str): Text to extract keyphrases from.

    Returns:
        list: List of top 10 candidate keyphrases.
    """
    out = []

    extractor = pke.unsupervised.MultipartiteRank()
    extractor.load_document(input=text, language='en')
    pos = {'PROPN'}
    stoplist = list(string.punctuation)
    stoplist += stopwords.words('english')
    extractor.candidate_selection(pos=pos)
    try:
        extractor.candidate_weighting(alpha=1.6,
                                      threshold=4,
                                      method='average')
    except:
        return out

    keyphrases = extractor.get_n_best(n=200)

    for key in keyphrases:
        out.append(key[0])

    return out

def extract_document_phrases(doc):
    """
    Extract noun phrases from a given document and return a list of the top 50 phrases.

    Args:
        doc (spacy.Doc): Document to extract noun phrases from.

    Returns:
        list: List of the top 50 noun phrases.
    """
    phrases = {}
    for np in doc.noun_chunks:
        phrase = np.text
        len_phrase = len(phrase.split())
        if len_phrase > 1:
            if phrase not in phrases:
                phrases[phrase] = 1
            else:
                phrases[phrase] += 1
    phrase_keys = sorted(phrases.keys(), key=lambda x: len(x), reverse=True)[:50]
    return phrase_keys


def edits(word):
    # Return all edits that are one edit away from the input word
    letters = 'abcdefghijklmnopqrstuvwxyz ' + string.punctuation
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def get_keywords(nlp,text,max_keywords,s2v,fdist,normalized_levenshtein,no_of_sentences):
    doc = nlp(text)
    max_keywords = int(max_keywords)

    keywords = get_nouns_multipartite(text)
    keywords = sorted(keywords, key=lambda x: fdist[x])
    keywords = filter_similar(keywords, max_keywords,normalized_levenshtein )

    phrase_keys = extract_document_phrases(doc)
    filtered_phrases = filter_similar(phrase_keys, max_keywords,normalized_levenshtein )

    total_phrases = keywords + filtered_phrases

    total_phrases_filtered = filter_similar(total_phrases, min(max_keywords, 2*no_of_sentences),normalized_levenshtein )


    answers = []
    for answer in total_phrases_filtered:
        if answer not in answers and check_distractors(answer,s2v):
            answers.append(answer)

    answers = answers[:max_keywords]
    return answers


def generate_questions_mcq(keyword_sent_mapping,device,tokenizer,model,sense2vec,normalized_levenshtein):
    batch_text = []
    answers = keyword_sent_mapping.keys()
    for answer in answers:
        txt = keyword_sent_mapping[answer]
        context = "context: " + txt
        text = context + " " + "answer: " + answer + " </s>"
        batch_text.append(text)

    encoding = tokenizer.batch_encode_plus(batch_text, pad_to_max_length=True, return_tensors="pt")
    print ("Generating...")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

    with torch.no_grad():
        outs = model.generate(input_ids=input_ids,
                              attention_mask=attention_masks,
                              max_length=150)

    output_array ={}
    output_array["questions"] =[]
    for index, val in enumerate(answers):
        individual_question ={}
        out = outs[index, :]
        dec = tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        Question = dec.replace("question:", "")
        Question = Question.strip()
        individual_question["question_statement"] = Question
        individual_question["question_type"] = "MCQ"
        individual_question["answer"] = val
        individual_question["id"] = index+1
        individual_question["options"], individual_question["options_algorithm"] = get_distractors_as_options(val, sense2vec)

        individual_question["options"] =  filter_similar(individual_question["options"], 10,normalized_levenshtein)
        index = 3
        individual_question["extra_options"]= individual_question["options"][index:]
        individual_question["options"] = individual_question["options"][:index]
        individual_question["context"] = keyword_sent_mapping[val]
     
        if len(individual_question["options"])>0:
            output_array["questions"].append(individual_question)

    return output_array


def random_choice():
    a = random.choice([0,1])
    return bool(a)
    
def decode_greedy(inp_ids, attn_mask, model, tokenizer):
    # Generate output using greedy decoding
    output_greedy_decode = model.generate(input_ids=inp_ids, attention_mask=attn_mask, max_length=256)
    # Decode output and capitalize the first letter
    question = tokenizer.decode(output_greedy_decode[0], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip().capitalize()
    return question

def decode_beam(inp_ids, attn_mask, model, tokenizer):
    # Generate multiple outputs using beam search decoding
    output_beam_search_decode = model.generate(input_ids=inp_ids, attention_mask=attn_mask, max_length=256, num_beams=10,
                                 num_return_sequences=3, no_repeat_ngram_size=2, early_stopping=True)
    # Decode each output and capitalize the first letter
    questions = [tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip().capitalize() for out in output_beam_search_decode]
    return questions

def check_distractors(word, s2v):
    # Check if MCQs are available for a given word
    word_preprocessed = word.translate(str.maketrans('', '', string.punctuation)).lower()
    sense = s2v.get_best_sense(word_preprocessed.replace(" ", "_"))
    return sense is not None


class GenerateQuestions:
    
    def __init__(self):
        # Initialize the tokenizer for T5
        self.tokenizer = T5Tokenizer.from_pretrained('t5new/tokenizer/')
        # self.tokenizer = AutoTokenizer.from_pretrained("nbroad/mt5-base-qgen")

        # Load the pre-trained T5 model
        self.model = T5ForConditionalGeneration.from_pretrained('t5New/newmodel')
        # self.model = AutoModelForSeq2SeqLM.from_pretrained("nbroad/mt5-base-qgen")

        # Set the device to use either CUDA or CPU depending on availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # Load the SpaCy model for English language processing
        self.nlp = spacy.load('en_core_web_sm')
        # Load the Sense2Vec model
        self.s2v = Sense2Vec().from_disk('s2v_old')
        # Load the Brown corpus and calculate the frequency distribution of its words
        self.fdist = FreqDist(brown.words())
        # Initialize the NormalizedLevenshtein algorithm for calculating string similarity
        self.normalized_levenshtein = NormalizedLevenshtein()
        # Set the seed for reproducibility
        self.set_seed(42)
   
    def set_seed(self, seed):
        # Set the seed for the random number generators in PyTorch and NumPy
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
            
    def predict_mcq(self, modelInputs):
        # Start measuring the execution time.
        start_time = time.time()
        # Extract the input text and the maximum number of questions from the modelInputs.
        input_text = modelInputs.get("input_text")
        max_questions = modelInputs.get("max_questions")
        # Tokenize the input text into sentences and join them into a single string.
        sentences = sentence_tokenize(input_text)
        modified_text = " ".join(sentences)
        # Extract the keywords from the modified text using the provided NLP tools.
        keywords = get_keywords(self.nlp, modified_text, max_questions, self.s2v, self.fdist, self.normalized_levenshtein, len(sentences))
        # Map each keyword to a list of up to three sentences that contain it.
        keyword_sentence_mapping = generate_keyword_sentences(keywords, sentences)
        for keyword in keyword_sentence_mapping:
            keyword_sentence_mapping[keyword] = " ".join(keyword_sentence_mapping[keyword][:3])
        # Generate multiple-choice questions from the keyword-sentence mappings.
        final_output = {}
        if keyword_sentence_mapping:
            try:
                generated_questions = generate_questions_mcq(keyword_sentence_mapping, self.device, self.tokenizer, self.model, self.s2v, self.normalized_levenshtein)
            except:
                # Ignore any errors during the question generation process.
                pass
            else:
                # Fill in the final output dictionary with the generated questions and some metadata.
                final_output["statement"] = modified_text
                final_output["questions"] = generated_questions["questions"]
                final_output["time_taken"] = time.time() - start_time
                # Free up any CUDA memory if running on a GPU.
                if torch.device == "cuda":
                    torch.cuda.empty_cache()
        # Return the final output dictionary.
        return final_output
        

class AnswerPredictor:
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        model = T5ForConditionalGeneration.from_pretrained('t5New/model')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # model.eval()
        self.device = device
        self.model = model
        self.set_seed(42)
    def set_seed(self,seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    def decode_greedy (inp_ids,attn_mask,model,tokenizer):
        output_greedy_decode = model.generate(input_ids=inp_ids, attention_mask=attn_mask, max_length=256)
        Question =  tokenizer.decode(output_greedy_decode[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
        return Question.strip().capitalize()
    def predict_answer(self,modelInputs):
        start = time.time()
        inp = {
            "input_text": modelInputs.get("input_text"),
            "input_question" : modelInputs.get("input_question")
        }
        context = inp["input_text"]
        question = inp["input_question"]
        input = "question: %s <s> context: %s </s>" % (question,context)
        encoding = self.tokenizer.encode_plus(input, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)
        output_greedy_decode = self.model.generate(input_ids=input_ids, attention_mask=attention_masks, max_length=256)
        Question =  self.tokenizer.decode(output_greedy_decode[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
        output = Question.strip().capitalize()
        return output


# qe= BoolQGen()
# output = qe.predict_boolq(modelInputs)
# pprint(output)
def text_summarizer(text):
    tokenizer = AutoTokenizer.from_pretrained("summarizer")
    model = AutoModelForSeq2SeqLM.from_pretrained("summarizer")
    inputs = tokenizer(text, max_length=1024, return_tensors="pt")
    summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=0)
    summarized_text = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return summarized_text

def generate_FIB_question(text, question_count):
    wrapper = textwrap.TextWrapper(width=150)
    word_list = wrapper.wrap(text=text)

    def tokenize_sentences(text):
        sentences = sent_tokenize(text)
        sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]
        return sentences
    
    sentences = tokenize_sentences(text)

    def get_noun_adj_verb(text):
        out=[]
        try:
            extractor = pke.unsupervised.MultipartiteRank()
            extractor.load_document(input=text,language='en')
            pos = {'VERB', 'ADJ', 'NOUN'}
            stoplist = list(string.punctuation)
            stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
            stoplist += stopwords.words('english')
            extractor.candidate_selection(pos=pos)
            extractor.candidate_weighting(alpha=1.1,
                                        threshold=0.75,
                                        method='average')
            keyphrases = extractor.get_n_best(n=question_count)
            for val in keyphrases:
                out.append(val[0])
        except:
            out = []
            traceback.print_exc()
        return out
    noun_verbs_adj = get_noun_adj_verb(text)
    def get_sentences_for_keyword(keywords, sentences):
        keyword_processor = KeywordProcessor()
        keyword_sentences = {}
        for word in keywords:
            keyword_sentences[word] = []
            keyword_processor.add_keyword(word)
        for sentence in sentences:
            keywords_found = keyword_processor.extract_keywords(sentence)
            for key in keywords_found:
                keyword_sentences[key].append(sentence)

        for key in keyword_sentences.keys():
            values = keyword_sentences[key]
            values = sorted(values, key=len, reverse=True)
            keyword_sentences[key] = values
        return keyword_sentences

    keyword_sentence_mapping_noun_verbs_adj = get_sentences_for_keyword(noun_verbs_adj, sentences)
    def get_fill_in_the_blanks(sentence_mapping):
        out={}
        blank_sentences = []
        processed = []
        keys=[]
        for key in sentence_mapping:
            if len(sentence_mapping[key])>0:
                sent = sentence_mapping[key][0]
                insensitive_sent = re.compile(re.escape(key), re.IGNORECASE)
                no_of_replacements =  len(re.findall(re.escape(key),sent,re.IGNORECASE))
                line = insensitive_sent.sub(' _________ ', sent)
                if (sentence_mapping[key][0] not in processed) and no_of_replacements<2:
                    blank_sentences.append(line)
                    processed.append(sentence_mapping[key][0])
                    keys.append(key)
        out["sentences"]=blank_sentences
        out["keys"]=keys
        return out


    fill_in_the_blanks = get_fill_in_the_blanks(keyword_sentence_mapping_noun_verbs_adj)
    output_json = [
        {"question": sentence, "answer": key}
    for sentence, key in zip(fill_in_the_blanks["sentences"], fill_in_the_blanks["keys"])
    ]
    print(output_json)
    return output_json
 

def generate_question(context, question_count):
    """Generate multiple-choice questions based on a given context.

    Args:
        context (str): The text from which to generate questions.
        question_count (int): The number of questions to generate.

    Returns:
        A tuple of lists containing the generated questions, correct answers, and distractors.
    """
    summarized_context = text_summarizer(context)
    classMCQ = GenerateQuestions()
    
    model_inputs = {
        "input_text": context,
        "max_questions": question_count
    }
    output = classMCQ.predict_mcq(model_inputs)
    questions_output = output['questions']
    for question in questions_output:
        while len(question['options']) < 3:
            question['options'].append('-')
    questions = [q['question_statement'] for q in questions_output]
    correct_answers = [a['answer'].capitalize() for a in questions_output]
    distractor1 = [d['options'][0] for d in questions_output]
    distractor2 = [d['options'][1] for d in questions_output]
    distractor3 = [d['options'][2] for d in questions_output]
    print(questions, correct_answers, distractor1, distractor2, distractor3)
    return summarized_context, questions, correct_answers, distractor1, distractor2, distractor3






