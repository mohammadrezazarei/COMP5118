from transformers import AutoModel
from transformers import AutoTokenizer
from sklearn.cluster import KMeans
import numpy as np
class Clustering:
    def __init__(self,k=50):
        self.model = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base")
        self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base")
        self.k = k
    def get_embedding(self, texts):
        self.model.eval()
        texts_lists = [texts[i:i + 10] for i in range(0, len(texts), 10)]
        features_mean_list = []
        for text_list in texts_lists:
            encoded_input = self.tokenizer(text_list, return_tensors='pt', padding = True)
            features = self.model(**encoded_input)
            features = features[0].detach().cpu().numpy() 
            features_mean = np.mean(features, axis=1) 
            features_mean_list.append(features_mean)
        return np.concatenate(features_mean_list,axis=0)
    def train_clustering(self, embeddings, random_state = 17):
        #embeddings = self.get_embedding(texts)
        clustering_model = KMeans(n_clusters=self.k, random_state=random_state)
        self.clustering_model = clustering_model.fit(embeddings)
    def predict(self,embeddings):
        #embeddings = self.get_embedding(texts)
        return self.clustering_model.predict(embeddings)

def identify_clusters(clusters,predicted,labels):
    '''identify the clusters for instances with wrong predictions'''
    clusters_list = []
    for i,label in enumerate(labels):
        if label != predicted[i]:
            clusters_list.append(clusters[i])
    return clusters_list

def sample_tweets_to_augment(cluster_num_to_tweets,clusters_to_augment, augment_num):
    '''Sample tweets from identified clusters for augmentation.'''
    import random
    tweets_to_augment = []
    augment_per_cluster = augment_num // len(clusters_to_augment)
    for cluster in clusters_to_augment:
        target_cluster_tweet = cluster_num_to_tweets[cluster]
        if len(target_cluster_tweet) > augment_per_cluster:
            tweets_to_augment += random.sample(target_cluster_tweet,augment_per_cluster)
        else:
            tweets_to_augment += [random.choice(target_cluster_tweet) for x in range(augment_per_cluster)]
          
    return tweets_to_augment