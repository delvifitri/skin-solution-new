
# coding: utf-8

import os
import re
from collections import defaultdict

class SentiStrengthID:
    list_negasi = []
    list_sentimen = []
    list_emoticon = []
    list_idiom = []
    list_booster = []
    dict_sentimen = defaultdict(lambda: 0)
    dict_emoticon = defaultdict(lambda: 0)
    dict_idiom = defaultdict(lambda: 0)
    dict_booster = defaultdict(lambda: 0)
    _scores = []
    _score = 0
    scores = []
    
    def __init__(self):
        self.list_negasi = [line.replace('\n','') for line in self._load('negatingword.txt')]
        self.list_sentimen = [line.replace('\n','').split(':') for line in self._load('sentiwords_id.txt')]
        self.list_emoticon = [line.replace('\n','').split(' | ') for line in self._load('emoticon_id.txt')]
        self.list_idiom = [line.replace('\n','').split(':') for line in self._load('idioms_id.txt')]
        self.list_booster = [line.replace('\n','').split(':') for line in self._load('boosterwords_id.txt')]
        
        for term in self.list_sentimen:
            self.dict_sentimen[term[0]] = int(term[1])
        
        for term in self.list_emoticon:
            self.dict_emoticon[term[0]] = int(term[1])
        
        for term in self.list_idiom:
            self.dict_idiom[term[0]] = int(term[1])
        
        for term in self.list_booster:
            self.dict_booster[term[0]] = int(term[1])
            
        self.re_newline = re.compile(r'\n')
        self.re_html_chars = re.compile(r'&[a-z]+;')
        self.re_url = re.compile(r'http\S+|www\S+')
        self.re_mention = re.compile(r'@\w+')
        self.re_hashtag = re.compile(r'#\w+')
        self.re_camel_case = re.compile(r'(?<!^)(?=[A-Z])')
        self.re_split_alpha1 = re.compile(r'([a-zA-Z])([^a-zA-Z\s!])')
        self.re_split_alpha2 = re.compile(r'([^a-zA-Z\s])([a-zA-Z])')
        self.re_whitespaces = re.compile(r'\s+')
        
        self.re_keep_emoticons = re.compile("|".join([re.escape(emoticon) for emoticon in self.dict_emoticon.keys()]))
        self.re_keep_chars = re.compile(r'[^a-zA-Z0-9\s\!\?\.\,]')
        
        self.re_multi_exclamation = re.compile(r'!{2,}')
        self.re_extra_chars = re.compile(r'([A-Za-z])\1{2,}')
        self.re_plural = re.compile(r'([A-Za-z]+)\-\1')
        
    def _load(self, filename):
        path = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(path, f'lexicon/{filename}'), 'r') as f:
            return f.read().splitlines()
    
    def _cek_negasi(self, p_term1, p_term2):
        if p_term1 in self.list_negasi or p_term2 in self.list_negasi or f'{p_term2} {p_term1}' in self.list_negasi:
            self._score = -self._score
    
    def _cek_booster(self, term):
        booster_score = self.dict_booster[term]
        self._score += booster_score if self._score > 0 else -booster_score
    
    def _cek_consecutive(self, p_term):
        if self._prev > 0 and self._score >= 3: self._score += 1
        if self._prev < 0 and self._score <= -3: self._score -= 1
    
    def _cek_idiom(self, bigram, trigram, i):
        bigram = ' '.join(bigram)
        trigram = ' '.join(trigram)
        idiom_score = self.dict_idiom[bigram] or self.dict_idiom[trigram]
        
        if idiom_score != 0:
            self._score = idiom_score
            self._prev = 0
    
    def _cek_penegasan(self, n_term):
        if self.re_multi_exclamation.search(n_term):
            self._score += 1 if self._score >= 3 else -3 if self._score <= -3 else 0
    
    def _remove_extra_char(self, term):
        return self.re_extra_chars.sub(r'\1',term)
    
    def _plural_to_singular(self, term):
        return self.re_plural.sub(r'\1',term)
    
    def _preprocess_symbols(self, text):
        emoticons_found = self.re_keep_emoticons.findall(text)

        for i, emoticon in enumerate(emoticons_found):
            text.replace(emoticon, f'EMOTICON#{i}#')
            
        for i, emoticon in enumerate(emoticons_found):
            text.replace(f'EMOTICON#{i}#', emoticon)

        return text
        
    
    def _preprocess(self, sentences):
        text = self.re_newline.sub(' ', sentences)
        text = self.re_html_chars.sub(' ', text)
        text = text.encode('ascii', 'ignore').decode('ascii')
        text = self.re_url.sub('', text)
        text = self.re_mention.sub('', text)
        text = self.re_hashtag.sub('', text)
        text = self.re_camel_case.sub(' ', text)
        text = self.re_split_alpha1.sub(r'\1 \2', text)
        text = self.re_split_alpha2.sub(r'\1 \2', text)
        text = text.lower()
        text = self._preprocess_symbols(text)
        text = self.re_whitespaces.sub(' ', text)
        return text
    
    def _normalize(self, score):
        return (score - (-7)) * (5 - 0) / (7 - (-7))        
    
    def score(self, review):
        text = ''
        sentences = self._preprocess(review)
        sentences = sentences.split('.')
        self.scores = []
        for sentence in sentences:
            self._scores = []
            terms = [term.strip(', ') for term in sentence.split()]
            terms_count = len(terms)
            self._prev = 0
            for i, term in enumerate(terms):
                is_extra_char = False
                plural = ''
                self._score = 0
                if self.re_extra_chars.search(term):
                    is_extra_char = True
                    term = self._remove_extra_char(term)
                if self.re_plural.search(term):
                    plural = term
                    term = self._plural_to_singular(term)
                term = term.lower()
                self._score = self.dict_sentimen[term]
                
                if self._score != 0 and i > 1:
                    self._cek_negasi(terms[i-1], terms[i-2])
                
                if self._score != 0:
                    if i > 0 and i <= terms_count -1: self._cek_booster(terms[i-1])
                    if i >= 0 and i < terms_count -1: self._cek_booster(terms[i+1])
                
                if i > 0 and i <= terms_count -1:
                    self._cek_idiom(terms[i-1:i+1], terms[i-1:i+2], i)
                
                if i > 0 and self._score !=0:
                    self._cek_consecutive(terms[i-1])
                
                if is_extra_char:
                    self._score += 1 if self._score > 0 else -1 if self._score < 0 else 2
                
                if i>= 0 and i < terms_count -1:
                    self._cek_penegasan(terms[i+1])
                                
                if self._score == 0:
                    self._score = self.dict_emoticon[term]
                
                text += f'{plural or term} '
                if self._prev == 0 and len(self._scores) > 0:
                    self._scores[-1] = 0
                self._scores.append(self._score)
                if self._score != 0:
                    text += f'[{self._score}] '
                self._prev = self._score
            self._scores = [score for score in self._scores if score != 0]
            if len(self._scores) > 0:
                self.scores.append(sum(self._scores) / len(self._scores))
        if len(self.scores) > 0:
            avg = sum(self.scores) / len(self.scores)
            return self._normalize(avg), text
        return 0, text