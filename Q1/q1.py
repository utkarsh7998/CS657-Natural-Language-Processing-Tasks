#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# ## Define cosine similarity

# In[2]:


def cosine_similarity(vector1, vector2):
    ans = np.dot(vector1, vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))
    return ans


# ## Define similarity above threshold checker function

# In[3]:


def are_they_Similar(score, th):
    if(score>th):
        return True ## Match, so no error
    else:
        return False #### Mismatch, so error = 1


# ## Make a global unique word dictionary

# In[4]:


unique_words = set()


# ## Importing Ground Truths and storing in dictionary and lists

# In[5]:


ground_truth = {}
i=1
with open('Word similarity/hindi.txt','r', encoding='utf-8') as file:
    for line in file.readlines():
        if(i==66):
            break
        i+=1
        x = line.split(',')
        term1,term2,similarity = x[0],x[1],x[2].strip('\n')
        ground_truth[(term1,term2)] = similarity
        unique_words.add(term1)
        unique_words.add(term2)


# In[6]:


# ground_truth


# ## Main function to calculate accuracy

# In[7]:


def get_accuracy(word2vector, ground_truth, th,modelname):
    errors = 0
    n = 0
    filename = 'Q1_'+modelname+'_similarity_'+str(int(th*10))+'.csv'
    f = open(filename,'w')
    for key in ground_truth.keys():
        actual_score = float(ground_truth[key])
        word1 = key[0]
        word2 = key[1]

        predicted_dist = cosine_similarity(np.array(word2vector[word1],dtype=float),np.array(word2vector[word2],dtype=float))
        predicted_score = predicted_dist*10
        
        actual_ans = are_they_Similar(float(ground_truth[key]), th*10)
        predicted_ans = are_they_Similar(predicted_score, th*10)
        
        label = -1
        if(predicted_ans!=actual_ans):
            errors += 1
            label = 1
        else:
            label = 0
        ## Writing output to file
        f.write(word1+'\t'+word2+'\t'+str(predicted_score)+'\t'+str(actual_score)+'\t'+str(label))
        f.write('\n')
#         print(word1,word2,predicted_score,ground_truth[key],label)
        n += 1
    
    accuracy = "{:.2f}".format(float(1-(errors/n)))
    
    f.write(str(accuracy))
    f.close()    
    return accuracy
        


# ## Reading Corpus downloaded from cflit website

# ## 1(a) Glove 50d model

# ### Part 1: Reading vector from glove file

# In[8]:


glove_w2v = {}
with open('hi/50/glove/hi-d50-glove.txt','r', encoding='utf-8') as file:
    n = len(unique_words)
    words_processed = 0
    for line in file:
        words = line.split(' ')
        term = words[0]
        if(term in unique_words):
            words[-1] = words[-1].strip('\n')
            vect = words[1:]
            glove_w2v[term] = vect
            words_processed += 1
            if(words_processed == n):
                break
        else:
            continue


# In[9]:


# glove_w2v


# ### Part 2: Getting accuracy at different thresholds

# In[10]:


print("Glove 50d")
th_vals = [0.4,0.5,0.6,0.7,0.8]
for th in th_vals:
    accuracy = get_accuracy(glove_w2v, ground_truth, th,"Glove 50d")
    print('threshold =',th,", accuracy = ",accuracy)


# ## 1(b) Glove 100d model

# ### Part 1: Reading vector from glove file

# In[11]:


glove_w2v = {}
with open('hi/100/glove/hi-d100-glove.txt','r', encoding='utf-8') as file:
    n = len(unique_words)
    words_processed = 0
    for line in file:
        words = line.split(' ')
        term = words[0]
        if(term in unique_words):
            words[-1] = words[-1].strip('\n')
            vect = words[1:]
            glove_w2v[term] = vect
            words_processed += 1
            if(words_processed == n):
                break
        else:
            continue


# ### Part 2: Getting accuracy at different thresholds

# In[12]:


print("Glove 100d")
th_vals = [0.4,0.5,0.6,0.7,0.8]
for th in th_vals:
    accuracy = get_accuracy(glove_w2v, ground_truth, th, "Glove 100d")
    print('threshold =',th,", accuracy = ", accuracy)


# ## 2(a)  Word2Vec CBOW 50d model

# ### Part 1: Reading vector from cbow file

# In[13]:


from gensim.models import Word2Vec
cbow_w2v = {}
model = Word2Vec.load('hi/50/cbow/hi-d50-m2-cbow.model')
n = len(unique_words)
words_processed = 0
for term in unique_words:        
    vect = model.wv[term]
    cbow_w2v[term] = vect
    words_processed += 1            
    if(words_processed == n):            
        break
    else:
        continue


# In[14]:


# cbow_w2v


# ### Part 2: Getting accuracy at different thresholds

# In[15]:


print("Word2Vec CBOW 50d")
th_vals = [0.4,0.5,0.6,0.7,0.8]
for th in th_vals:
    accuracy = get_accuracy(cbow_w2v, ground_truth, th, 'Word2Vec CBOW 50d')
    print('threshold =',th,", accuracy = ", accuracy)


# ## 2(b)  Word2Vec CBOW 100d model

# ### Part 1: Reading vector from cbow file

# In[16]:


from gensim.models import Word2Vec
cbow_w2v = {}
model = Word2Vec.load('hi/100/cbow/hi-d100-m2-cbow.model')
n = len(unique_words)
words_processed = 0
for term in unique_words:        
    vect = model.wv[term]
    cbow_w2v[term] = vect
    words_processed += 1            
    if(words_processed == n):            
        break
    else:
        continue


# ### Part 2: Getting accuracy at different thresholds

# In[17]:


print("Word2Vec CBOW 100d")
th_vals = [0.4,0.5,0.6,0.7,0.8]
for th in th_vals:
    ac = get_accuracy(cbow_w2v, ground_truth, th,"Word2Vec CBOW 100d")
    print('threshold =',th,", accuracy = ", ac)


# ## 3(a)  Word2Vec Skipgram 50d model

# ### Part 1: Reading vector from cbow file

# In[18]:


from gensim.models import Word2Vec
sg_w2v = {}
model = Word2Vec.load('hi/50/sg/hi-d50-m2-sg.model')
n = len(unique_words)
words_processed = 0
for term in unique_words:        
    vect = model.wv[term]
    sg_w2v[term] = vect
    words_processed += 1            
    if(words_processed == n):            
        break
    else:
        continue


# ### Part 2: Getting accuracy at different thresholds

# In[19]:


print("Word2Vec skipgram 50d")
th_vals = [0.4,0.5,0.6,0.7,0.8]
for th in th_vals:
    ac = get_accuracy(sg_w2v, ground_truth, th, "Word2Vec skipgram 50d")
    print('threshold =',th,", accuracy = ", ac)


# ## 3(b)  Word2Vec Skipgram 100d model

# ### Part 1: Reading vector from cbow file

# In[20]:


from gensim.models import Word2Vec
sg_w2v = {}
model = Word2Vec.load('hi/100/sg/hi-d100-m2-sg.model')
n = len(unique_words)
words_processed = 0
for term in unique_words:        
    vect = model.wv[term]
    sg_w2v[term] = vect
    words_processed += 1
#     print(words_processed)
    if(words_processed == n):            
        break
    else:
        continue


# ### Part 2: Getting accuracy at different thresholds

# In[21]:


print("Word2Vec Skipgram 100d")
th_vals = [0.4,0.5,0.6,0.7,0.8]
for th in th_vals:
    ac = get_accuracy(sg_w2v, ground_truth, th,"Word2Vec Skipgram 100d")
    print('threshold =',th,", accuracy = ", ac)


# ## 4(a)  Fast-text 50d model

# ### Part 1: Reading vector from cbow file

# In[22]:


from gensim.models import FastText
ft_w2v = {}
model = FastText.load('hi/50/fasttext/hi-d50-m2-fasttext.model')
n = len(unique_words)
words_processed = 0
for term in unique_words:        
    vect = model.wv[term]
    ft_w2v[term] = vect
    words_processed += 1
#     print(words_processed)
    if(words_processed == n):            
        break
    else:
        continue


# ### Part 2: Getting accuracy at different thresholds

# In[23]:


print("FastText 50d")
th_vals = [0.4,0.5,0.6,0.7,0.8]
for th in th_vals:
    ac = get_accuracy(ft_w2v, ground_truth, th, "FastText 50d")
    print('threshold =',th,", accuracy = ", ac)


# ## 4(b)  Fast-text 100d model

# ### Part 1: Reading vector from cbow file

# In[24]:


from gensim.models import FastText
ft_w2v = {}
model = FastText.load('hi/100/fasttext/hi-d100-m2-fasttext.model')
n = len(unique_words)
words_processed = 0
for term in unique_words:        
    vect = model.wv[term]
    ft_w2v[term] = vect
    words_processed += 1
#     print(words_processed)
    if(words_processed == n):            
        break
    else:
        continue


# ### Part 2: Getting accuracy at different thresholds

# In[25]:


print("FastText 100d")
th_vals = [0.4,0.5,0.6,0.7,0.8]
for th in th_vals:
    ac  = get_accuracy(ft_w2v, ground_truth, th, "FastText 100d")
    print('threshold =',th,", accuracy = ", ac)


# In[26]:


print("completed question 1")

