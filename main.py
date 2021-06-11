#Rami bou Abboud
#40043011
#Comp 472 Summer 2021
#A2 Zixi


import re
import csv
import numpy as np
import requests
from bs4 import BeautifulSoup
import pandas as pd


season = []
episodes = []
airdates = []
seasons = []
reviews = []

reviews_page = []
all_comment = []
positive_words = []
negative_words = []
reviews_ratings = []
positive_count = 0
negative_count = 0
vocabulary = []
all_words = []
word_count_p = []
word_count_n = []
deleted_words = []
positive_words_vocabulary = []
negative_words_vocabulary = []

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
            if int(user_rating) >= 8:
                positive_count += 1
            else:
                negative_count += 1
            reviews_ratings.append(user_rating)
        except:
            reviews_ratings.append(0)



#splits positive with negative comments
for i in range(len(all_comment)):
    if int(reviews_ratings[i]) >= 8:
        positive_words.append(all_comment[i])
    elif int(reviews_ratings[i]) < 8 and int(reviews_ratings[i]) != 0:
        negative_words.append(all_comment[i])



positive_words = ''.join(positive_words)
positive_words = positive_words.lower().split()
negative_words = ''.join(negative_words)
negative_words = negative_words.lower().split()

#keeps only alpha numerics
for i in range(len(positive_words)):
    positive_words[i] = re.sub('[\W_]+', '', positive_words[i])

for i in range(len(negative_words)):
    negative_words[i] = re.sub('[\W_]+', '', negative_words[i])

all_words = np.concatenate((positive_words, negative_words))

for x in all_words:
    if x not in vocabulary:
        vocabulary.append(x)

for x in positive_words:
    if x not in positive_words_vocabulary:
        positive_words_vocabulary.append(x)

for x in negative_words:
    if x not in negative_words_vocabulary:
        negative_words_vocabulary.append(x)

# counts words in positive from vocabulary + 1 smoothing
for i in range(len(vocabulary)):
    posi_cnt = positive_words.count(vocabulary[i])   #count of total word in positive array
    word_count_p.append(posi_cnt + 1) #smoothing and add to array
    if(posi_cnt > 400): #can tweak this number for accuracy
        deleted_words.append(vocabulary[i])

# counts words in negative from vocabulary + 1 smoothing
for i in range(len(vocabulary)):
    nega_cnt = negative_words.count(vocabulary[i])   #count of total word in positive array
    word_count_n.append(nega_cnt + 1) #smoothing and add to array
    if(nega_cnt > 400 and vocabulary[i] not in deleted_words): #can tweak this number for accuracy
        deleted_words.append(vocabulary[i])

#removes delete words from array
for x in deleted_words:
    word_count_p.pop(vocabulary.index(x))
    word_count_n.pop(vocabulary.index(x))
    vocabulary.remove(x)



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
    for i in range(int(len(vocabulary) / 2)):
        file.write('No.' + str(i + 1) + '  ' + str(vocabulary[i]) + '\n')
        file.write(
            str(word_count_p[i]) + ', ' + str('{:.5f}'.format(word_count_p[i] / len(all_words))) + ', ' + str(
                word_count_n[i]) + ', ' + str('{:.5f}'.format(word_count_n[i] / len(all_words))) + '\n')






#tweak set: all words above 400 dupes removed for better accuracy
print('Sample of how many reviews?(distinct): ' + str(len(all_comment)))
print('\nvocabulary size(distinct): ' + str(len(vocabulary)))
print('Each element means the numbers of times in positive array: ' + str(len(word_count_p)))
print('Each element means the numbers of times in negative array: ' + str(len(word_count_n)))
print('positive words vocabulary(distinct): ' + str(len(positive_words_vocabulary)))
print('negative words vocabulary(distinct): ' + str(len(negative_words_vocabulary)))
print('\npositive words size(with duplicates): ' + str(len(positive_words))) #words in positive comments
print('negative words size(with duplicates): ' + str(len(negative_words))) #words in negative comments
print('total words(with duplicates): ' + str(len(all_words)))  #total numbers of words