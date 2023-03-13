
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
import numpy as np 
import pandas as pd 
import time
import random
import spacy
import zipfile
import os
import json
from sense2vec import Sense2Vec
import requests
from collections import OrderedDict
import string
import pke
import nltk
from nltk import FreqDist
nltk.download('brown')
nltk.download('stopwords')
nltk.download('popular')
from nltk.corpus import stopwords
from nltk.corpus import brown
from similarity.normalized_levenshtein import NormalizedLevenshtein
from nltk.tokenize import sent_tokenize
from flashtext import KeywordProcessor







def greedy_decoding (inp_ids,attn_mask,model,tokenizer):
  greedy_output = model.generate(input_ids=inp_ids, attention_mask=attn_mask, max_length=256)
  Question =  tokenizer.decode(greedy_output[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
  return Question.strip().capitalize()


def beam_search_decoding (inp_ids,attn_mask,model,tokenizer):
  beam_output = model.generate(input_ids=inp_ids,
                                 attention_mask=attn_mask,
                                 max_length=256,
                               num_beams=10,
                               num_return_sequences=3,
                               no_repeat_ngram_size=2,
                               early_stopping=True
                               )
  Questions = [tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True) for out in
               beam_output]
  return [Question.strip().capitalize() for Question in Questions]


def topkp_decoding (inp_ids,attn_mask,model,tokenizer):
  topkp_output = model.generate(input_ids=inp_ids,
                                 attention_mask=attn_mask,
                                 max_length=256,
                               do_sample=True,
                               top_k=40,
                               top_p=0.80,
                               num_return_sequences=3,
                                no_repeat_ngram_size=2,
                                early_stopping=True
                               )
  Questions = [tokenizer.decode(out, skip_special_tokens=True,clean_up_tokenization_spaces=True) for out in topkp_output]
  return [Question.strip().capitalize() for Question in Questions]



def MCQs_available(word,s2v):
    word = word.replace(" ", "_")
    sense = s2v.get_best_sense(word)
    if sense is not None:
        return True
    else:
        return False


def edits(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz '+string.punctuation
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def sense2vec_get_words(word,s2v):
    output = []

    word_preprocessed =  word.translate(word.maketrans("","", string.punctuation))
    word_preprocessed = word_preprocessed.lower()

    word_edits = edits(word_preprocessed)

    word = word.replace(" ", "_")

    sense = s2v.get_best_sense(word)
    most_similar = s2v.most_similar(sense, n=15)

    compare_list = [word_preprocessed]
    for each_word in most_similar:
        append_word = each_word[0].split("|")[0].replace("_", " ")
        append_word = append_word.strip()
        append_word_processed = append_word.lower()
        append_word_processed = append_word_processed.translate(append_word_processed.maketrans("","", string.punctuation))
        if append_word_processed not in compare_list and word_preprocessed not in append_word_processed and append_word_processed not in word_edits:
            output.append(append_word.title())
            compare_list.append(append_word_processed)


    out = list(OrderedDict.fromkeys(output))

    return out

def get_options(answer,s2v):
    distractors =[]

    try:
        distractors = sense2vec_get_words(answer,s2v)
        if len(distractors) > 0:
            print(" Sense2vec_distractors successful for word : ", answer)
            return distractors,"sense2vec"
    except:
        print (" Sense2vec_distractors failed for word : ",answer)


    return distractors,"None"

def tokenize_sentences(text):
    sentences = [sent_tokenize(text)]
    sentences = [y for x in sentences for y in x]
    sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]
    return sentences


def get_sentences_for_keyword(keywords, sentences):
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


def is_far(words_list,currentword,thresh,normalized_levenshtein):
    threshold = thresh
    score_list =[]
    for word in words_list:
        score_list.append(normalized_levenshtein.distance(word.lower(),currentword.lower()))
    if min(score_list)>=threshold:
        return True
    else:
        return False

def filter_phrases(phrase_keys,max,normalized_levenshtein ):
    filtered_phrases =[]
    if len(phrase_keys)>0:
        filtered_phrases.append(phrase_keys[0])
        for ph in phrase_keys[1:]:
            if is_far(filtered_phrases,ph,0.8,normalized_levenshtein ):
                filtered_phrases.append(ph)
            if len(filtered_phrases)>=max:
                break
    return filtered_phrases


def get_nouns_multipartite(text):
    out = []

    extractor = pke.unsupervised.MultipartiteRank()
    extractor.load_document(input=text, language='en')
    pos = {'PROPN', 'NOUN'}
    stoplist = list(string.punctuation)
    stoplist += stopwords.words('english')
    extractor.candidate_selection(pos=pos)
    try:
        extractor.candidate_weighting(alpha=1.1,
                                      threshold=0.75,
                                      method='average')
    except:
        return out

    keyphrases = extractor.get_n_best(n=10)

    for key in keyphrases:
        out.append(key[0])

    return out


def get_phrases(doc):
    phrases={}
    for np in doc.noun_chunks:
        phrase =np.text
        len_phrase = len(phrase.split())
        if len_phrase > 1:
            if phrase not in phrases:
                phrases[phrase]=1
            else:
                phrases[phrase]=phrases[phrase]+1

    phrase_keys=list(phrases.keys())
    phrase_keys = sorted(phrase_keys, key= lambda x: len(x),reverse=True)
    phrase_keys=phrase_keys[:50]
    return phrase_keys



def get_keywords(nlp,text,max_keywords,s2v,fdist,normalized_levenshtein,no_of_sentences):
    doc = nlp(text)
    max_keywords = int(max_keywords)

    keywords = get_nouns_multipartite(text)
    keywords = sorted(keywords, key=lambda x: fdist[x])
    keywords = filter_phrases(keywords, max_keywords,normalized_levenshtein )

    phrase_keys = get_phrases(doc)
    filtered_phrases = filter_phrases(phrase_keys, max_keywords,normalized_levenshtein )

    total_phrases = keywords + filtered_phrases

    total_phrases_filtered = filter_phrases(total_phrases, min(max_keywords, 2*no_of_sentences),normalized_levenshtein )


    answers = []
    for answer in total_phrases_filtered:
        if answer not in answers and MCQs_available(answer,s2v):
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


    print ("Running model for generation")
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
        individual_question["options"], individual_question["options_algorithm"] = get_options(val, sense2vec)

        individual_question["options"] =  filter_phrases(individual_question["options"], 10,normalized_levenshtein)
        index = 3
        individual_question["extra_options"]= individual_question["options"][index:]
        individual_question["options"] = individual_question["options"][:index]
        individual_question["context"] = keyword_sent_mapping[val]
     
        if len(individual_question["options"])>0:
            output_array["questions"].append(individual_question)

    return output_array

def generate_normal_questions(keyword_sent_mapping,device,tokenizer,model):  #for normal one word questions
    batch_text = []
    answers = keyword_sent_mapping.keys()
    for answer in answers:
        txt = keyword_sent_mapping[answer]
        context = "context: " + txt
        text = context + " " + "answer: " + answer + " </s>"
        batch_text.append(text)

    encoding = tokenizer.batch_encode_plus(batch_text, pad_to_max_length=True, return_tensors="pt")


    print ("Running model for generation")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

    with torch.no_grad():
        outs = model.generate(input_ids=input_ids,
                              attention_mask=attention_masks,
                              max_length=150)

    output_array ={}
    output_array["questions"] =[]
    
    for index, val in enumerate(answers):
        individual_quest= {}
        out = outs[index, :]
        dec = tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        Question= dec.replace('question:', '')
        Question= Question.strip()

        individual_quest['Question']= Question
        individual_quest['Answer']= val
        individual_quest["id"] = index+1
        individual_quest["context"] = keyword_sent_mapping[val]
        
        output_array["questions"].append(individual_quest)
        
    return output_array

def random_choice():
    a = random.choice([0,1])
    return bool(a)
    


class QGen:
    
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained('t5new/tokenizer/')
        model = T5ForConditionalGeneration.from_pretrained('t5New/newmodel')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        self.device = device
        self.model = model
        self.nlp = spacy.load('en_core_web_sm')

        self.s2v = Sense2Vec().from_disk('s2v_old')

        self.fdist = FreqDist(brown.words())
        self.normalized_levenshtein = NormalizedLevenshtein()
        self.set_seed(42)
        
    def set_seed(self,seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
    def predict_mcq(self, payload):
        start = time.time()
        inp = {
            "input_text": payload.get("input_text"),
            "max_questions": payload.get("max_questions")
        }

        text = inp['input_text']
        sentences = tokenize_sentences(text)
        joiner = " "
        modified_text = joiner.join(sentences)


        keywords = get_keywords(self.nlp,modified_text,inp['max_questions'],self.s2v,self.fdist,self.normalized_levenshtein,len(sentences) )


        keyword_sentence_mapping = get_sentences_for_keyword(keywords, sentences)

        for k in keyword_sentence_mapping.keys():
            text_snippet = " ".join(keyword_sentence_mapping[k][:3])
            keyword_sentence_mapping[k] = text_snippet

   
        final_output = {}

        if len(keyword_sentence_mapping.keys()) == 0:
            return final_output
        else:
            try:
                generated_questions = generate_questions_mcq(keyword_sentence_mapping,self.device,self.tokenizer,self.model,self.s2v,self.normalized_levenshtein)

            except:
                return final_output
            end = time.time()

            final_output["statement"] = modified_text
            final_output["questions"] = generated_questions["questions"]
            final_output["time_taken"] = end-start
            
            if torch.device=='cuda':
                torch.cuda.empty_cache()
                
            return final_output
    
    def predict_shortq(self, payload):
        inp = {
            "input_text": payload.get("input_text"),
            "max_questions": payload.get("max_questions")
        }

        text = inp['input_text']
        sentences = tokenize_sentences(text)
        joiner = " "
        modified_text = joiner.join(sentences)


        keywords = get_keywords(self.nlp,modified_text,inp['max_questions'],self.s2v,self.fdist,self.normalized_levenshtein,len(sentences) )


        keyword_sentence_mapping = get_sentences_for_keyword(keywords, sentences)
        
        for k in keyword_sentence_mapping.keys():
            text_snippet = " ".join(keyword_sentence_mapping[k][:3])
            keyword_sentence_mapping[k] = text_snippet

        final_output = {}

        if len(keyword_sentence_mapping.keys()) == 0:
            print('ZERO')
            return final_output
        else:
            
            generated_questions = generate_normal_questions(keyword_sentence_mapping,self.device,self.tokenizer,self.model)
            print(generated_questions)

            
        final_output["statement"] = modified_text
        final_output["questions"] = generated_questions["questions"]
        
        if torch.device=='cuda':
            torch.cuda.empty_cache()

        return final_output
            
  
    def paraphrase(self,payload):
        start = time.time()
        inp = {
            "input_text": payload.get("input_text"),
            "max_questions": payload.get("max_questions")
        }

        text = inp['input_text']
        num = inp['max_questions']
        
        self.sentence= text
        self.text= "paraphrase: " + self.sentence + " </s>"

        encoding = self.tokenizer.encode_plus(self.text,pad_to_max_length=True, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)

        beam_outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            max_length= 50,
            num_beams=50,
            num_return_sequences=num,
            no_repeat_ngram_size=2,
            early_stopping=True
            )
        final_outputs =[]
        for beam_output in beam_outputs:
            sent = self.tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
            if sent.lower() != self.sentence.lower() and sent not in final_outputs:
                final_outputs.append(sent)
        
        output= {}
        output['Question']= text
        output['Count']= num
        output['Paraphrased Questions']= final_outputs
        
        for i, final_output in enumerate(final_outputs):
            print("{}: {}".format(i, final_output))

        if torch.device=='cuda':
            torch.cuda.empty_cache()
        
        return output


class BoolQGen:
       
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_boolean_questions')
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

    def random_choice(self):
        a = random.choice([0,1])
        return bool(a)
    

    def predict_boolq(self,payload):
        start = time.time()
        inp = {
            "input_text": payload.get("input_text"),
            "max_questions": payload.get("max_questions")
        }

        text = inp['input_text']
        num= inp['max_questions']
        sentences = tokenize_sentences(text)
        joiner = " "
        modified_text = joiner.join(sentences)
        answer = self.random_choice()
        form = "truefalse: %s passage: %s </s>" % (modified_text, answer)

        encoding = self.tokenizer.encode_plus(form, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)

        output = beam_search_decoding (input_ids, attention_masks,self.model,self.tokenizer)
        if torch.device=='cuda':
            torch.cuda.empty_cache()
        
        final= {}
        final['Text']= text
        final['Count']= num
        final['Boolean Questions']= output
            
        return final
            
class AnswerPredictor:
          
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        model = T5ForConditionalGeneration.from_pretrained('Parth/boolean')
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

    def greedy_decoding (inp_ids,attn_mask,model,tokenizer):
        greedy_output = model.generate(input_ids=inp_ids, attention_mask=attn_mask, max_length=256)
        Question =  tokenizer.decode(greedy_output[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
        return Question.strip().capitalize()

    def predict_answer(self,payload):
        start = time.time()
        inp = {
            "input_text": payload.get("input_text"),
            "input_question" : payload.get("input_question")
        }

        context = inp["input_text"]
        question = inp["input_question"]
        input = "question: %s <s> context: %s </s>" % (question,context)

        encoding = self.tokenizer.encode_plus(input, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)
        greedy_output = self.model.generate(input_ids=input_ids, attention_mask=attention_masks, max_length=256)
        Question =  self.tokenizer.decode(greedy_output[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
        output = Question.strip().capitalize()

        return output


# qe= BoolQGen()
# output = qe.predict_boolq(payload)
# pprint(output)


def generate_question(context, questionCount):
    qg = QGen()
    payload = {
            "input_text": context,
            "max_questions": questionCount
        }
    output = qg.predict_mcq(payload)
    questionsoutput = output['questions']

    for question in questionsoutput:
      while len(question['options']) < 3:
          question['options'].append('-')
   
    questions = [questions['question_statement'] for questions in questionsoutput]
    correctAnswers = [answers['answer'].capitalize() for answers in questionsoutput]
    distractor1 = [distractor['options'][0] for distractor in questionsoutput]
    distractor2 = [distractor['options'][1] for distractor in questionsoutput]
    distractor3 = [distractor['options'][2] for distractor in questionsoutput]

    print(questions, correctAnswers, distractor1, distractor2, distractor3)
    return questions, correctAnswers, distractor1, distractor2, distractor3
  # np = get_keywords(context, summary_text, questionCount)
    # questions = []
    # correctAnswers = []
    # distractor1 = []
    # distractor2 = []
    # distractor3 = []
    # for answer in np:
    #     ques = get_question(summary_text, answer, question_model, question_tokenizer)
    #     distractors = get_distractors(answer.capitalize(), ques, s2v, sentence_transformer_model, 40, 0.2)
    #     questions.append(ques)
    #     correctAnswers.append(answer.capitalize())
    #     if len(distractors) > 0:
    #         for i, distractor in enumerate(distractors[:4]):
    #           if i == 0:
    #               distractor1.append(distractor)
    #           elif i == 1:
    #               distractor2.append(distractor)
    #           elif i == 2:
    #               distractor3.append(distractor)
    #     else:
    #         for i, distractor in enumerate(distractors[:4]):
    #           if i == 0:
    #               distractor1.append("Couldnt Generate")
    #           elif i == 1:
    #               distractor2.append("Couldnt Generate")
    #           elif i == 2:
    #               distractor3.append("Couldnt Generate")