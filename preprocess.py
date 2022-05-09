import re
from nltk.corpus import stopwords
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


def replace_all_blank(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    res = " ".join(text.split()) 
    return res


def codefilter(htmlstr):
    s = re.sub(r'(<code>)(\n|.)*?(</code>)', ' ', htmlstr, re.S)
    return s

def htmlfilter(htmlstr):
 
    re_cdata = re.compile('//<!\[CDATA\[[^>]*//\]\]>', re.I) 
    re_script = re.compile('<\s*script[^>]*>[^<]*<\s*/\s*script\s*>', re.I)  
    re_style = re.compile('<\s*style[^>]*>[^<]*<\s*/\s*style\s*>', re.I) 
    re_br = re.compile('<br\s*?/?>') 
    re_h = re.compile('</?\w+[^>]*>')  
    re_comment = re.compile('<!--[^>]*-->')  
    re_url = re.compile(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%|-|_)*\b', re.MULTILINE)

    s = re_cdata.sub('', htmlstr)  
    s = re_script.sub('', s)  
    s = re_style.sub('', s) 
    s = re_br.sub(' ', s)
    s = re_h.sub('', s)  
    s = re_comment.sub('', s)  
    s = re_url.sub('', s)
    s = re.sub(r'\\n(\B|\b)', ' ',s)
    return s

def replaceCharEntity(htmlstr):
    CHAR_ENTITIES = {'nbsp': ' ', '160': ' ',
                     'lt': '<', '60': '<',
                     'gt': '>', '62': '>',
                     'amp': '&', '38': '&',
                     'quot': '"', '34': '"', }

    re_charEntity = re.compile(r'&#?(?P<name>\w+);')
    sz = re_charEntity.search(htmlstr)
    while sz:
        entity = sz.group()  
        key = sz.group('name')  
        try:
            htmlstr = re_charEntity.sub(CHAR_ENTITIES[key], htmlstr, 1)
            sz = re_charEntity.search(htmlstr)
        except KeyError:
            htmlstr = re_charEntity.sub('', htmlstr, 1)
            sz = re_charEntity.search(htmlstr)
    return htmlstr

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def preprocess(str):
    str = re.sub(r"won't", "will not", str)
    str = re.sub(r"can't", "can not", str)
    str = re.sub(r"n't", " not", str)
    str = re.sub(r"'ve", " have", str)
    str = re.sub(r"'ll", " will", str)
    str = re.sub(r"'re", " are", str)

    str = str.lower()

    str = replace_all_blank(str)

    tokens = word_tokenize(str)

    tagged_sent = pos_tag(tokens) 

    wnl = WordNetLemmatizer()
    lemmas_sent = []
    for tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))

    english_stopwords = stopwords.words("english")
    words_stopwordsremoved = []
    for word in lemmas_sent:
        if word not in english_stopwords:
            words_stopwordsremoved.append(word)
    result = ''
    for word in words_stopwordsremoved:
        result += ' ' + word
    return result.strip()


def processbody(str):
    str = replaceCharEntity(str)
    str = codefilter(str)
    str = htmlfilter(str)
    str = preprocess(str)
    return str


if __name__ == '__main__':
    df = pd.read_csv('all_questions.csv')
    for index, row in df.iterrows():
        title = str(row['title'])
        body = str(row['body'])
        df.at[index, 'title'] = processbody(title)
        df.at[index, 'body'] = processbody(body)
        print(df.at[index, 'title'])
    df.to_csv("processed_questions.csv")
