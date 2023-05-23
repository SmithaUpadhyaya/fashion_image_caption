import unicodedata
import string
import re

table = str.maketrans('','',string.punctuation)

#Data cleaning- lower casing, removing puntuations and words containing numbers
def cleaning_text(title):

    desc = title

    #remove extra space between the words
    #desc = remove_extra_space(desc)

    #1. remove extra space between the words 
    #2. converts to lowercase
    #3. expand contractions
    #4. remove punctuation/special
    #5. remove accented characters like â, î, or ô
    desc = [remove_accent_chars(remove_punctuation(expand_contractions(convert_lower(word)))) for word in desc.split()] 

    #remove repeated words

    #remove words with numbers in them
    #desc = [word for word in desc if(word.isalpha())]

    #join it back to string
    desc =  ' '.join(desc)
    
    return desc

# expand contractions(e.g can't = can + not)
def expand_contractions(text):

    #return contractions.fix(text)
    return text
    
#correcting mis-spelled words, return all possible spelling possible 
#install pyspellchecker 
def spell_check(text):
    """
    from spellchecker import SpellChecker
 
    spell = SpellChecker()
    
    # find those words that may be misspelled
    misspelled = spell.unknown(["cmputr", "watr", "study", "wrte"])
    
    for word in misspelled:
        # Get the one `most likely` answer
        print(spell.correction(word))
    
        # Get a list of `likely` options
        print(spell.candidates(word))
    """
    pass

#remove accented characters like â, î, or ô
def remove_accent_chars(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

#remove extra space between the words
def remove_extra_space(text):
    return [word for word in text.split()]

#converts to lowercase
def convert_lower(text):   
    #return [word.lower() for word in text]
    return text.lower()
    
#remove punctuation/special chars(!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~) from each word 
def remove_punctuation(text):
    
    #return [word.translate(table) for word in text]
    return text.translate(table)

def check_punctuation(text):

    pattern = "[" + string.punctuation + "]"
    match = re.findall(pattern,text)

    #if match:
    #    return 1
    #else:
    #    return 0
    return match

def clean_color_feature(text):

    #remove punctuation
    pattern = '[' + string.punctuation + ']'
    text = re.sub(pattern, '', text)

    ## convert to lower
    text = text.lower()

    ## remove any extra spaces in the text
    text = " ".join(text.split()) 

    ## clean numeric in text
    #Remove numeric with some measure 
    # Most common type of measurement cm: centimeter, inch, mm: millimeter,foot, k: killometer

    pattern = r'\d+[a-z]+' #This will consider eg. 32mm, 18k, 23cm etc
    text = re.sub(pattern, '', text)
    
    #Remove numeric pattern like "1 Year", "11 Year"
    pattern = r'[\d+\s+]+years'
    text = re.sub(pattern, '', text)
    pattern = r'[\d+\s+]+year'
    text = re.sub(pattern, '', text)

    #Remove numeric pattern like 'only 1 left', 'Only 2 left'
    pattern = r'only[\s+\d+]+lefts'
    text = re.sub(pattern, '', text)
    pattern = r'only[\s+\d+]+left'
    text = re.sub(pattern, '', text)

    #Remove numeric values
    pattern = r'\d+'
    text = re.sub(pattern, '', text)

    #remove word with one or two char other
    #pattern = r'\b\w{1,2}\s*' #tried to find the right patter for string like "A B", "A COLOR", "B COLOR B"
    #text = re.sub(pattern, '', text)
    text = text.split()
    text = [word.strip() for word in text if len(word.strip()) > 2]

    ## again remove any extra spaces in the text in case addation space get added.
    if len(text) > 0:
        text = " ".join(text)
    else:
        text = "" 
    
    return text.strip()


def number_2_word(n):
 
    arr = ['zero','one','two','three','four','five','six','seven','eight','nine']

    # If all the digits are encountered return blank string
    if(n == 0):
        return ""
     
    else:
        # compute spelling for the last digit
        small_ans = arr[n%10]
 
        # keep computing for the previous digits and add the spelling for the last digit
        ans = number_2_word(int(n/10)) + small_ans + " "
     
    # Return the final answer
    return ans

def clean_title_feature(text):

    #remove punctuation
    pattern = '[' + string.punctuation + ']'
    text = re.sub(pattern, '', text)

    ## convert to lower
    text = text.lower()

    ## remove any extra spaces in the text
    text = " ".join(text.split()) 

    ## clean numeric in text
    #Remove numeric with some measure 
    # Most common type of measurement cm: centimeter, inch, mm: millimeter,foot, k: killometer

    pattern = r'\d+[a-z]+' #This will consider eg. 32mm, 18k, 23cm etc
    text = re.sub(pattern, '', text)
    
    pattern = r'(\d+\s+(inch|cm|mm|k))' #This will consider eg. 32 inch, 501 k, 55 cm etc
    text = re.sub(pattern, '', text)

    #Remove numeric pattern like "woodstock 6 7 8 plus iphone case", "facet iphone 7 plus 8 plus case"
    pattern = r'[\d+\s+]+plus'
    text = re.sub(pattern, '', text)
    
    #Remove numeric pattern like "shannon iphone 11 11 pro 11 pro max folio case"
    pattern = r'[\d+\s+]+pro' 
    text = re.sub(pattern, '', text)

    #Remove numeric value of type like year or digit more the 2 char
    pattern = r'\d{2,}'
    text = re.sub(pattern, '', text)

    #Remove numeric values
    pattern = r'\d+'
    text = re.sub(pattern, lambda x: number_2_word(int(x.group())), text)

    #remove word with one or two char other then stopwords
    stopwords = {'an', 'be', 'do', 'of', 'is', 's', 'am', 'or', 'as', 'we', 'me','up', 'to', 'no', 'at', 'in', 'on', 'so', 'he', 'i', 't', 'if', 'my', 'a', 'by', 'it' }
    pattern = r'\b[\w]{1,2}\s'
    fiter = re.finditer(pattern, text)

    replace_index = []

    try:

        while(True):

            idx = fiter.__next__().span()            

            if not text[idx[0]:idx[1]].strip() in stopwords:
                replace_index.append(idx)            

    except StopIteration:
        #end of the iter reached
        pass
    
    for i in range(len(replace_index), 0, -1):
                
        idx = replace_index[i - 1]
        text = text[0:idx[0]] + text[idx[1]:]

    #remove duplicate words
    # help reference: https://www.geeksforgeeks.org/remove-duplicate-words-from-sentence-using-regular-expression/ did not work since it only remove duplicate if the word repeat next to each other
    # did not work
    #pattern = r'\b(\w+)(?:\W+\1\b)+' 
    #text = re.sub(pattern, r'\1', text, flags = re.IGNORECASE)

    #only keep the last occurance. as observed most on the time the repeated word is the product type and we have observed that product type usually the last word in the title of the item
    split_text = text.split()
    split_text.reverse()
    text = []
    for word in split_text:
        if word in text:
            continue
        else:
            text.append(word.strip())

    ## join the string back after removing duplicate
    text.reverse()
    text = " ".join(text) 
    
    return text.strip()