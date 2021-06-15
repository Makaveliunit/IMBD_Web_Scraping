#Rami bou Abboud
#40043011
#Comp 472 Summer 2021
#A2 Zixi

import math
import re
import numpy as np
import requests
from bs4 import BeautifulSoup
import pandas as pd
import random as rn

from idna import unicode

season = []
episodes = []
airdates = []
seasons = []
reviews = []

titles_list = []
reviews_page = []
all_comment = []
positive_words = []
negative_words = []
reviews_ratings = []
all_words = []
word_count_p = []
word_count_n = []
deleted_words = []
positive_titles = []
negative_titles = []

#inserts html of each season in an array
for x in range(1,6):
    season.append(requests.get("https://www.imdb.com/title/tt3032476/episodes?season="+ str(x)))


# cycles through the seasons
for i in range(len(season)):
    soup = BeautifulSoup(season[i].content, 'html.parser')

    #returns all the divs we need to retrieve info
    movie_containers = soup.find_all('div', class_ = 'list_item')



    #fetches all the info needed and stores them in arrays
    for x in range(len(movie_containers)):
        episode = movie_containers[x]
        title = episode.div.find('a')['title']
        episodes.append(title)
        airdate = episode.find('div', class_ = 'airdate').text.strip(' \n\t')
        airdates.append(airdate[-4:])
        seas_ep = episode.div.a.text.strip(' \n\t')
        seasons.append(seas_ep[1])
        review_link = 'https://www.imdb.com' + episode.div.a['href'] + 'reviews'
        reviews.append(review_link)



#sets up the data with panda before write to csv
data = pd.DataFrame({
    "Name": episodes,
    "Season": seasons,
    "Review Link": reviews,
    "Year": airdates,
})
data.to_csv('data')


#stores every review html
for x in range(len(reviews)):
    reviews_page.append(requests.get(reviews[x]))

# cycles through the seasons
for i in range(len(reviews_page)):
    soup = BeautifulSoup(reviews_page[i].content, 'html.parser')



    #returns all the divs we need to retrieve info
    review_containers = soup.find_all('div', class_ = 'lister-item')
    for x in range(len(review_containers)):
        user_review = review_containers[x].find('div', class_='text show-more__control').text
        all_comment.append(user_review)
        try:
            user_rating = review_containers[x].find('span', class_='rating-other-user-rating')
            user_rating = user_rating.span.text
            reviews_ratings.append(user_rating)
        except:
            reviews_ratings.append(0)


    title_containers = soup.find_all('div', class_ = 'lister-item')
    for x in range(len(title_containers)):
        titles = title_containers[x].a.text
        titles_list.append(titles)



#splits positive with negative comments
for i in range(len(all_comment)):
    if int(reviews_ratings[i]) >= 8:
        positive_words.append(all_comment[i].replace('.', ' '))
        positive_titles.append(titles_list[i])
    elif int(reviews_ratings[i]) < 8 and int(reviews_ratings[i]) != 0:
        negative_words.append(all_comment[i].replace('.', ' '))
        negative_titles.append(titles_list[i])



positive_words = ' '.join(positive_words)
positive_words = positive_words.lower().split()
negative_words = ' '.join(negative_words)
negative_words = negative_words.lower().split()

training_data_positive = []
training_data_negative = []
training_data_vocabulary = []
testing_data_positive = []
testing_data_negative = []

#keeps only alpha numerics and take half of each list
for i in range(len(positive_words)):
    positive_words[i] = re.sub('[\W_]+', ' ', positive_words[i])
    if i < int(len(positive_words)/2):
        training_data_positive.append(positive_words[i].replace(' ',''))
    else:
        testing_data_positive.append(positive_words[i].replace(' ',''))

for i in range(len(negative_words)):
    negative_words[i] = re.sub('[\W_]+', ' ', negative_words[i])
    if i < int(len(negative_words)/2):
        training_data_negative.append(negative_words[i].replace(' ',''))
    else:
        testing_data_negative.append(negative_words[i].replace(' ',''))


#put both in distinct vocabulary
for i in range(len(training_data_positive)):
    if training_data_positive[i] not in training_data_vocabulary:
        training_data_vocabulary.append(training_data_positive[i])

for i in range(len(training_data_negative)):
    if training_data_negative[i] not in training_data_vocabulary:
        training_data_vocabulary.append(training_data_negative[i])



# counts words in positive from vocabulary + 1 smoothing
for i in range(len(training_data_vocabulary)):
    posi_cnt = training_data_positive.count(training_data_vocabulary[i])   #count of total word in positive array
    word_count_p.append(posi_cnt + 1) #smoothing and add to array
    if(posi_cnt > 200 and training_data_vocabulary[i] not in deleted_words): #can tweak this number for accuracy
        deleted_words.append(training_data_vocabulary[i])

# counts words in negative from vocabulary + 1 smoothing
for i in range(len(training_data_vocabulary)):
    nega_cnt = training_data_negative.count(training_data_vocabulary[i])   #count of total word in positive array
    word_count_n.append(nega_cnt + 1) #smoothing and add to array
    if(nega_cnt > 200 and training_data_vocabulary[i] not in deleted_words): #can tweak this number for accuracy
       deleted_words.append(training_data_vocabulary[i])

with open('stopword.txt') as f:
    lines = f.read().splitlines()
    for x in lines:
        deleted_words.append(x)


#removes delete words from array
for x in deleted_words:
    if x in training_data_vocabulary:
        word_count_p.pop(training_data_vocabulary.index(x))
        word_count_n.pop(training_data_vocabulary.index(x))
        training_data_vocabulary.remove(x)
    while x in testing_data_positive:
        testing_data_positive.remove(x)
    while x in testing_data_negative:
        testing_data_negative.remove(x)


#writes delete words in csv
deleted_list = pd.DataFrame({
    "Deleted words": deleted_words,
})
deleted_list.to_csv('remove')

#now we need to set the training and testing sets to 50% each
#we will only be computing half of the data array and the rest will remain for testing purposes



#MODEL
#writes on txt No. WordName(wi)
#Frequency in Positive, Conditional probability of P(wi|positive),
with open('model.txt', 'w') as file:
    for i in range(int(len(training_data_vocabulary))):
        file.write('No.' + str(i + 1) + '  ' + str(training_data_vocabulary[i]) + '\n')
        file.write(
            str(word_count_p[i]) + ', ' + str('{:.5f}'.format(word_count_p[i] / len(training_data_positive))) + ', ' + str(
                word_count_n[i]) + ', ' + str('{:.5f}'.format(word_count_n[i] / len(training_data_negative))) + '\n')


#now we can start testing data
with open('probabilities.txt', 'w') as file:
    for i in range(len(training_data_vocabulary)):
       file.write('\nPositive: ' + str(word_count_p[i]) + '   ' + str(training_data_vocabulary[i]) + '   ' + str((math.log10(0.82) + math.log10(word_count_p[i] / len(training_data_positive))))
           + '      Negative: ' + str(word_count_n[i]) + '   ' + str(training_data_vocabulary[i]) + '   ' + str((math.log10(0.18) + math.log10(word_count_n[i] / len(training_data_negative)))))

correct_results = 0
correct_results2 = 0
predictions_results = []
prediction_results2 = []

def compute(wi):
    positive_score = 0
    negative_score = 0
    positive_word_count = 1  #smoothing
    negative_word_count = 1  #smoothing
    if wi in training_data_vocabulary:
        positive_word_count = (word_count_p[training_data_vocabulary.index(wi)])  #smoothing
        negative_word_count = (word_count_n[training_data_vocabulary.index(wi)]) #smoothing
    positive_score = math.log10(0.82) + math.log10(positive_word_count/len(training_data_positive))
    negative_score = math.log10(0.18) + math.log10(negative_word_count/len(training_data_negative))
    print(str(positive_score) + '    ' + str(positive_word_count) + '   ' + str(negative_score) + '   ' +  str(negative_word_count))
    if positive_score >= negative_score or (positive_word_count + negative_word_count) == 2:
        return 'Positive'
    else:
        return 'Negative'



title_neg_prob = []
title_pos_prob = []

#returns whether phrase is positive or negative
def compute_titles(wi):
    positive_score = 0
    negative_score = 0
    positive_prob = 0
    negative_prob = 0
    positive_word_count = 1  #smoothing
    negative_word_count = 1  #smoothing
    wi = re.sub('[\W_]+', ' ', wi)
    wi = wi.lower().split()
    for x in wi:
        if x in training_data_vocabulary:
             positive_word_count = (word_count_p[training_data_vocabulary.index(x)])  #smoothing
             negative_word_count = (word_count_n[training_data_vocabulary.index(x)]) #smoothing
        positive_score += math.log10(0.82) + math.log10(positive_word_count/len(training_data_positive))
        negative_score += math.log10(0.18) + math.log10(negative_word_count/len(training_data_negative))
        positive_prob += positive_word_count/len(training_data_positive)
        negative_prob += negative_word_count/len(training_data_negative)
    title_pos_prob.append('{:.5f}'.format(positive_prob))
    title_neg_prob.append('{:.5f}'.format(negative_prob))
    if positive_score >= negative_score:
        return 'Positive'
    else:
        return 'Negative'

def isEqual(str1, str2):
    if str1 == str2:
        return 'Right'
    else:
        return 'Wrong'


with open('results.txt', 'w') as file:
    for i in range(len(positive_titles)):
        result = compute_titles(positive_titles[i])
        if result == 'Positive':
            correct_results2 += 1
        try:
            file.write('No.' + str(i + 1) + '   ' + str(positive_titles[i]))
            file.write(str(title_pos_prob[i]) + ', ' + str(title_neg_prob[i]) + ', ' + str(result) + ', Positive, ' + isEqual(result, 'Positive') + '\n\n')
        except: continue
    for i in range(len(negative_titles)):
        result = compute_titles(negative_titles[i])
        if result == 'Negative':
            correct_results2 += 1
        try:
            file.write('No.' + str(i + 1) + '   ' + str(negative_titles[i]))
            file.write(str(title_pos_prob[i]) + ', ' + str(title_neg_prob[i]) + ', ' + str(result) + ', Negative, ' + isEqual(result, 'Negative') + '\n\n')
        except: continue
    file.write('Prediction accuracy = ' + str('{:.2f}'.format((correct_results2/(len(positive_titles) + len(negative_titles)))*100) + '%'))


#predictions and test
for i in range(len(testing_data_positive)):
    result = compute(testing_data_positive[i])
    if result == 'Positive':
        correct_results += 1
    print('No.' + str(i + 1) + '   ' + testing_data_positive[i])
    print('my prediction: ' + str(result) + '\n')

print('ALL THE NEGATIVES BELOW #########################')
for i in range(len(testing_data_negative)):
    result = compute(testing_data_negative[i])
    if result == 'Negative':
        correct_results += 1
    print('No.' + str(i + 1) + '   ' + testing_data_negative[i])
    print('my prediction: ' + str(result) + '\n')

print('Prediction accuracy = ' + str('{:.2f}'.format((correct_results/(len(testing_data_positive) + len(testing_data_negative)))*100) + '%'))

#function that trims down the data to [num] length
def trim_down(num):
    for x in training_data_vocabulary[:]:
        if len(x) < num:
            word_count_p.pop(training_data_vocabulary.index(x))
            word_count_n.pop(training_data_vocabulary.index(x))
            training_data_vocabulary.remove(x)
            while x in testing_data_positive:
                testing_data_positive.remove(x)
            while x in testing_data_negative:
                testing_data_negative.remove(x)
            while x in training_data_positive:
                training_data_positive.remove(x)
            while x in training_data_negative:
                training_data_negative.remove(x)

#function that trims up the data to [num] length
def trim_up(num):
    for x in training_data_vocabulary[:]:
        if len(x) > num:
            word_count_p.pop(training_data_vocabulary.index(x))
            word_count_n.pop(training_data_vocabulary.index(x))
            training_data_vocabulary.remove(x)
            while x in testing_data_positive:
                testing_data_positive.remove(x)
            while x in testing_data_negative:
                testing_data_negative.remove(x)
            while x in training_data_positive:
                training_data_positive.remove(x)
            while x in training_data_negative:
                training_data_negative.remove(x)

#trims the list down to 3+ characters only per words
trim_down(3)

# NEW MODEL
# writes on txt No. WordName(wi)
# Frequency in Positive, Conditional probability of P(wi|positive),
with open('length-model.txt', 'w') as file:
    file.write('\n\n\n length <= 2 XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX \n\n\n')
    for i in range(int(len(training_data_vocabulary))):
        file.write('No.' + str(i + 1) + '  ' + str(training_data_vocabulary[i]) + '\n')
        file.write(
            str(word_count_p[i]) + ', ' + str(
                '{:.5f}'.format(word_count_p[i] / len(training_data_positive))) + ', ' + str(
                word_count_n[i]) + ', ' + str(
                '{:.5f}'.format(word_count_n[i] / len(training_data_negative))) + '\n')

correct_results2 = 0
title_neg_prob = []
title_pos_prob = []

# predictions and test
with open('length-results.txt', 'w') as file:
    file.write('\n\n\n length <= 2 XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX \n\n\n')
    for i in range(len(testing_data_positive)):
        result = compute_titles(testing_data_positive[i])
        if result == 'Positive':
            correct_results2 += 1
        try:
            file.write('No.' + str(i + 1) + '   ' + str(testing_data_positive[i]) + '\n')
            file.write(str(title_pos_prob[i]) + ', ' + str(title_neg_prob[i]) + ', ' + str(
                result) + ', Positive, ' + isEqual(result, 'Positive') + '\n\n')
        except:
            continue
    for i in range(len(testing_data_negative)):
        result = compute_titles(testing_data_negative[i])
        if result == 'Negative':
            correct_results2 += 1
        try:
            file.write('No.' + str(i + 1) + '   ' + str(testing_data_negative[i]) + '\n')
            file.write(str(title_pos_prob[i]) + ', ' + str(title_neg_prob[i]) + ', ' + str(
                result) + ', Negative, ' + isEqual(result, 'Negative') + '\n\n')
        except:
            continue
    file.write('Prediction accuracy = ' + str('{:.2f}'.format(
        (correct_results2 / (len(testing_data_positive) + len(testing_data_negative))) * 100) + '%'))

trim_down(5)

#NEW MODEL
#writes on txt No. WordName(wi)
#Frequency in Positive, Conditional probability of P(wi|positive),
with open('length-model.txt', 'a') as file:
    file.write('\n\n\n length <= 4 XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX \n\n\n')
    for i in range(int(len(training_data_vocabulary))):
        file.write('No.' + str(i + 1) + '  ' + str(training_data_vocabulary[i]) + '\n')
        file.write(
            str(word_count_p[i]) + ', ' + str('{:.5f}'.format(word_count_p[i] / len(training_data_positive))) + ', ' + str(
                word_count_n[i]) + ', ' + str('{:.5f}'.format(word_count_n[i] / len(training_data_negative))) + '\n')

correct_results2 = 0
title_neg_prob = []
title_pos_prob = []

#predictions and test
with open('length-results.txt', 'a') as file:
    file.write('\n\n\n length <= 4 XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX \n\n\n')
    for i in range(len(testing_data_positive)):
        result = compute_titles(testing_data_positive[i])
        if result == 'Positive':
            correct_results2 += 1
        try:
            file.write('No.' + str(i + 1) + '   ' + str(testing_data_positive[i]) + '\n')
            file.write(str(title_pos_prob[i]) + ', ' + str(title_neg_prob[i]) + ', ' + str(result) + ', Positive, ' + isEqual(result, 'Positive') + '\n\n')
        except: continue
    for i in range(len(testing_data_negative)):
        result = compute_titles(testing_data_negative[i])
        if result == 'Negative':
            correct_results2 += 1
        try:
            file.write('No.' + str(i + 1) + '   ' + str(testing_data_negative[i]) + '\n')
            file.write(str(title_pos_prob[i]) + ', ' + str(title_neg_prob[i]) + ', ' + str(result) + ', Negative, ' + isEqual(result, 'Negative') + '\n\n')
        except: continue
    file.write('Prediction accuracy = ' + str('{:.2f}'.format((correct_results2/(len(testing_data_positive) + len(testing_data_negative)))*100) + '%'))

trim_up(8)

#NEW MODEL
#writes on txt No. WordName(wi)
#Frequency in Positive, Conditional probability of P(wi|positive),
with open('length-model.txt', 'a') as file:
    file.write('\n\n\n length >= 9  XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX \n\n\n')
    for i in range(int(len(training_data_vocabulary))):
        file.write('No.' + str(i + 1) + '  ' + str(training_data_vocabulary[i]) + '\n')
        file.write(
            str(word_count_p[i]) + ', ' + str('{:.5f}'.format(word_count_p[i] / len(training_data_positive))) + ', ' + str(
                word_count_n[i]) + ', ' + str('{:.5f}'.format(word_count_n[i] / len(training_data_negative))) + '\n')

correct_results2 = 0
title_neg_prob = []
title_pos_prob = []

#predictions and test
with open('length-results.txt', 'a') as file:
    file.write('\n\n\n length >= 9  XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX \n\n\n')
    for i in range(len(testing_data_positive)):
        result = compute_titles(testing_data_positive[i])
        if result == 'Positive':
            correct_results2 += 1
        try:
            file.write('No.' + str(i + 1) + '   ' + str(testing_data_positive[i]) + '\n')
            file.write(str(title_pos_prob[i]) + ', ' + str(title_neg_prob[i]) + ', ' + str(result) + ', Positive, ' + isEqual(result, 'Positive') + '\n\n')
        except: continue
    for i in range(len(testing_data_negative)):
        result = compute_titles(testing_data_negative[i])
        if result == 'Negative':
            correct_results2 += 1
        try:
            file.write('No.' + str(i + 1) + '   ' + str(testing_data_negative[i]) + '\n')
            file.write(str(title_pos_prob[i]) + ', ' + str(title_neg_prob[i]) + ', ' + str(result) + ', Negative, ' + isEqual(result, 'Negative') + '\n\n')
        except: continue
    file.write('Prediction accuracy = ' + str('{:.2f}'.format((correct_results2/(len(testing_data_positive) + len(testing_data_negative)))*100) + '%'))

#tweak set: all words above 400 dupes removed for better accuracy
print('Sample of how many reviews?(distinct): ' + str(len(all_comment)))
print('Each element means the numbers of times in positive array: ' + str(len(word_count_p)))
print('Each element means the numbers of times in negative array: ' + str(len(word_count_n)))
print('\npositive words size(with duplicates): ' + str(len(positive_words))) #words in positive comments
print('negative words size(with duplicates): ' + str(len(negative_words))) #words in negative comments
print('training + data count(with duplicates): ' + str(len(training_data_positive)))
print('training - data count(with duplicates): ' + str(len(training_data_negative)))
print('testing + data count(with duplicates): ' + str(len(testing_data_positive)))
print('testing - data count(with duplicates): ' + str(len(testing_data_negative)))
print('training vocabulary(DISTINCT): ' + str(len(training_data_vocabulary)))

