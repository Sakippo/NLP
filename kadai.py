import nltk
import re
import string
import random
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

###各種ダウンロード
#nltk.download('averaged_perceptron_tagger')
#nltk.download('punkt')
#nltk.download('twitter_samples')
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('wordnet')
#nltk.download('omw-1.4')
###

def clean_tweets(tweet):

    #tweetから内容だけを抽出
    tweet = re.sub(r'&.*', '', tweet) #html特殊文字の削除
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet) #リンクを削除
    tweet = re.sub(r'#', '', tweet) #ハッシュタグの削除
    tweet = re.sub(r'\@[A-Za-z0-9_]*','', tweet) #アカウント名の削除

    #tweetを単語に分解
    tokenizer = nltk.tokenize.TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    #単語からstopwords,punctuation,絵文字を削除
    stopwords = nltk.corpus.stopwords.words('english')
    emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
    ])
    emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
    ])
    emoticons = emoticons_happy.union(emoticons_sad)

    stemmer = nltk.stem.PorterStemmer()

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords and word not in emoticons and word not in string.punctuation):  
            tweets_clean.append(word)
    
    #レマタイズを行う
    tweets_clean = lemmatize(nltk.pos_tag(tweets_clean))
    tweet = ""
    for word in tweets_clean:
        tweet += word + " "
    
    return tweet

def lemmatize(sentence):
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in sentence:
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence


def main():
    tweet_samples = nltk.corpus.twitter_samples
    positive_tweets = tweet_samples.strings('positive_tweets.json')
    negative_tweets = tweet_samples.strings('negative_tweets.json')

    # positive tweets set
    positive_tweets_set = []
    for tweet in positive_tweets:
        positive_tweets_set.append([clean_tweets(tweet), 'pos'])    
 
    # negative tweets set
    negative_tweets_set = []
    for tweet in negative_tweets:
        negative_tweets_set.append([clean_tweets(tweet), 'neg'])

    #データセットをシャッフルする
    random.shuffle(positive_tweets_set)
    random.shuffle(negative_tweets_set)

    #訓練集合とテスト集合の準備
    train_set = positive_tweets_set[1000:] + negative_tweets_set[1000:]
    test_set = positive_tweets_set[:1000] + negative_tweets_set[:1000]
    
    train_x = [l[0] for l in train_set]
    train_y = [l[1] for l in train_set]
    test_x =  [l[0] for l in test_set]
    test_y =  [l[1] for l in test_set]

    tf = TfidfVectorizer()
    train_tf = tf.fit_transform(train_x)
    test_tf = tf.transform(test_x)

    #学習
    naive_bayes_classifier = MultinomialNB()
    naive_bayes_classifier.fit(train_tf, train_y)

    # テスト
    y_pred = naive_bayes_classifier.predict(test_tf)
    accuracy = metrics.accuracy_score(test_y, y_pred)
    print(accuracy) 

    #手動入力文章でのテスト
    #明らかにネガティブな文章入力例
    custom_tweet_1 = "It was a disaster. No link with the originals. Ridiculous. Nuff said.."
    custom_tweet_1 = clean_tweets(custom_tweet_1)
    custom_tweet_1_pred = naive_bayes_classifier.predict(tf.transform([custom_tweet_1]))
    print("Negative tweet:")
    print(custom_tweet_1_pred) 

    #明らかにポジティブな文章入力例
    custom_tweet_2 = "I saw this movie and loved it - favorite thing I saw. It was so moving & real. The entire cast was terrific.."
    custom_tweet_2 = clean_tweets(custom_tweet_2)
    custom_tweet_2_pred = naive_bayes_classifier.predict(tf.transform([custom_tweet_2]))
    print("Positive tweet:")
    print(custom_tweet_2_pred) 

if __name__ == '__main__':
    main()


