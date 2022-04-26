import numpy as np
import torch
import argparse
import pickle
import os

from sklearn.model_selection import train_test_split
import torch.nn.functional as F

from torch.utils.data import DataLoader
from transformers import AdamW

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer




from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from Augmenter import Augmenter
from Clustering import Clustering
from Clustering import identify_clusters, sample_tweets_to_augment


os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser(description='Wise Augmentation Application')
parser.add_argument('--dataset', default='./data', help='dataset location')
parser.add_argument('--save_dir', default='./results', help='Save directory location')
parser.add_argument('--cluster_num', default=50, type = int, help='Number of clusters')
parser.add_argument('--task_name', default='offensive', help='Name of task. It can be offensive or hate')
parser.add_argument('--load_clusters', default=0, type = int, help='Load clusters if they have been previously trained and saved')
parser.add_argument('--dl_word_embeddings', default=0, type = int, help='Download pretrained word embeddings')
parser.add_argument('--augmentation_model', default='fasttext', help='Augmentation_model')
parser.add_argument('--augmentation_percentage', default=0.2, type = float, help='Augmentation_percentage')
parser.add_argument('--wise_augmentation', default=1, type = int, help='If wise augmentation should be performed')

def serializeObject(object_,file_name):
    file_object = open(file_name,'wb')
    pickle.dump(object_, file_object,protocol = 2)
    file_object.close()
    return
def deserializeObject(file_name):
    file_object = open(file_name,'rb')
    object_ = pickle.load(file_object)
    file_object.close() 
    return object_
def read_file(file):
    lst = []
    with open(file,'r') as f:
        for readline in f: 
            line_strip = readline.strip()
            lst.append(line_strip)
    return lst

def read_dataset(tweet_file,label_file):
    tweets = read_file(tweet_file)
    labels = np.array([int(i) for i in read_file(label_file)], dtype = np.int64)
    return tweets,labels

def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }

def get_prediction_single_tweet(text,model):
    inputs = tokenizer(text, padding=True, return_tensors="pt").to("cuda")
    outputs = model(**inputs)
    probs = outputs[0].softmax(1)
    return torch.argmax(probs).item()

def get_prediction_set(text_set,labels,model):
    predicted = [get_prediction_single_tweet(tweet,model) for tweet in text_set]
    return classification_accuracy(np.array(predicted),labels)


def classification_accuracy(pred,labels):
    cm = confusion_matrix(labels, pred)
    accuracy_per_class = cm.diagonal()/cm.sum(axis=1)
    accuracy_all, accuracy_0, accuracy_1 = cm.trace()/cm.sum(), accuracy_per_class[0], accuracy_per_class[1]
    return accuracy_all, accuracy_0, accuracy_1



def get_classifier_model(model_name,tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return model, tokenizer

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)
    
if __name__ == "__main__":
    args = parser.parse_args()
    dataset_dir = args.dataset
    save_dir = args.save_dir
    task_name = args.task_name
    load_clusters = args.load_clusters
    dl_word_embeddings = args.dl_word_embeddings
    augmentation_model = args.augmentation_model
    augmentation_percentage = args.augmentation_percentage
    wise_augmentation = args.wise_augmentation
    
    if dl_word_embeddings:
        from nlpaug.util.file.download import DownloadUtil
        DownloadUtil.download_fasttext(model_name='wiki-news-300d-1M', dest_dir='.') # Download fasttext model
        print('FasttText pre-traned model downloaded.')
    
    
    
    root_address = dataset_dir + '/' + task_name + '/'
    
    save_origin = save_dir + '/' + task_name + '/'
    
    
    if not os.path.isdir(root_address):
        os.makedirs(root_address)
    
    
    tweets_train, labels_train = read_dataset(root_address+'train_text.txt', root_address+'train_labels.txt')
    tweets_test, labels_test = read_dataset(root_address+'test_text.txt', root_address+'test_labels.txt')
    tweets_validation, labels_validation = read_dataset(root_address+'val_text.txt', root_address+'val_labels.txt')


    augmentor = Augmenter()

    if wise_augmentation==0:
        if augmentation_model == 'fasttext':
            tweets_augmented = augmentor.augmentation_word(tweets_train, aug_p=augmentation_percentage)
        elif augmentation_model == 'spelling':
            tweets_augmented = augmentor.augmentation_spelling(tweets_train, aug_p=augmentation_percentage)
        elif augmentation_model == 'gpt2':
            tweets_augmented = augmentor.augmentation_sentence(tweets_train, model_type='gpt2')
        tweets_train = tweets_train + tweets_augmented
        labels_train = np.concatenate((labels_train,labels_train))

    model_name = "cardiffnlp/twitter-roberta-base"
    model, tokenizer = get_classifier_model(model_name, model_name)
    print('Pre-trained RoBERTa loaded successfully!')
    
    train_encodings = tokenizer(tweets_train,  padding=True)
    val_encodings = tokenizer(tweets_validation, padding=True)
    
    train_dataset = Dataset(train_encodings, labels_train)
    valid_dataset = Dataset(val_encodings, labels_validation)
    
    
    cluster_dir = root_address+ 'clusters/'
    if not os.path.isdir(cluster_dir):
        os.makedirs(cluster_dir)
    
    if load_clusters:
        train_clusters = deserializeObject(cluster_dir+'train_clusters')
        valid_clusters = deserializeObject(cluster_dir+'valid_clusters')
        print('Clusters loaded successfully!')
    else:
        cc = Clustering()
        embeddings_train = cc.get_embedding(tweets_train)
        cc.train_clustering(embeddings_train)
        embeddings_val = cc.get_embedding(tweets_validation)
        train_clusters = cc.predict(embeddings_train)
        valid_clusters = cc.predict(embeddings_val) 
        serializeObject(train_clusters,cluster_dir+'train_clusters')
        serializeObject(valid_clusters,cluster_dir+'valid_clusters')
        print('Clusters are trained and saved successfully!')
    
    model_name_folder = 'twitter_roberta_base/'
    if wise_augmentation == 0:
        version = 'augmented_original/'
    else:
        version = 'augmented_wise/'
    version += augmentation_model + '_' + str(augmentation_percentage) + '/' 
    save_dir_log = save_origin + model_name_folder + version + 'log/'
    save_dir_model = save_origin + model_name_folder + version + 'model/'
    if not os.path.isdir(save_dir_log):
        os.makedirs(save_dir_log)
    if not os.path.isdir(save_dir_model):
        os.makedirs(save_dir_model)
        
        
    cluster_num_to_tweets = {}
    for i,cluster_num in enumerate(train_clusters):
        cluster_tweets = cluster_num_to_tweets.get(cluster_num,[])
        cluster_tweets.append((tweets_train[i],labels_train[i]))
        cluster_num_to_tweets[cluster_num] = cluster_tweets
        
        
    device = torch.device('cuda')
    model.to(device)
    
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers = 6, pin_memory = True)
    optim = AdamW(model.parameters(), lr=5e-5)
    
    
    best_val_acc = 0
    print('Training is starting!')
    for epoch in range(10):
        model.train()
        epoch_loss = 0
        
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
        model.eval()
        
        
        #do augmentation
        if wise_augmentation == 1:
            predicted = [get_prediction_single_tweet(tweet,model) for tweet in tweets_validation]
            clusters_to_augment = identify_clusters(valid_clusters,predicted,labels_validation)
            sampled_tweets_tuples =   sample_tweets_to_augment(cluster_num_to_tweets,clusters_to_augment,len(tweets_train))
            sampled_tweets = [tw[0] for tw in sampled_tweets_tuples]
            sampled_labels = [tw[1] for tw in sampled_tweets_tuples]
            
            if augmentation_model == 'fasttext':
                tweets_augmented = augmentor.augmentation_word(sampled_tweets, aug_p=augmentation_percentage)
            elif augmentation_model == 'spelling':
                tweets_augmented = augmentor.augmentation_spelling(sampled_tweets, aug_p=augmentation_percentage)
            elif augmentation_model == 'gpt2':
                tweets_augmented = augmentor.augmentation_sentence(sampled_tweets, model_type='gpt2')

            
        
            tweets_train_new = tweets_train + tweets_augmented
            labels_new = np.concatenate((labels_train, np.array(sampled_labels, dtype = np.int64)))
            train_encodings = tokenizer(tweets_train_new, padding=True)
            train_dataset = Dataset(train_encodings, labels_new)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers = 6, pin_memory = True)
    
        
        val_acc = get_prediction_set(tweets_validation,labels_validation,model)[0]
        test_acc = get_prediction_set(tweets_test,labels_test,model)[0]
        print(f'Iter {epoch+1}: | Loss: {epoch_loss/len(train_loader):.5f} | Vall acc: {val_acc:.5f} | Test acc: {test_acc:.5f}')
        with open(save_dir_log+'log.txt', 'a') as f:
            f.write('Epoch {:d} | Loss: {:f} | Vall acc {:f} | Test acc {:f}\n'.format(epoch+1, epoch_loss/len(train_loader), val_acc, test_acc))
        if val_acc > best_val_acc:
            with open(save_dir_log+'best_log.txt', 'a') as f:
                f.write('Epoch {:d} | Loss: {:f} | Vall acc {:f} | Test acc {:f}\n'.format(epoch+1, epoch_loss/len(train_loader), val_acc, test_acc))
            model.save_pretrained(save_dir_model+'model_best')
            best_val_acc = val_acc
    