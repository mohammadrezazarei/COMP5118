import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc

class Augmenter:
    def augmentation_word(self,tweets, model_type='fasttext', action = 'substitute', aug_p=0.2, num_thread = 8):
        aug = naw.WordEmbsAug(model_type=model_type, model_path='wiki-news-300d-1M.vec',action = action, aug_p = aug_p)
        augmented_tweets = aug.augment(tweets, num_thread = num_thread)
        return augmented_tweets
    
    def augmentation_sentence(self,tweets, model_type='gpt2'):
        aug = nas.ContextualWordEmbsForSentenceAug(model_path = model_type,device='cuda')
        augmented_tweets = []
        for tweet in tweets:
            augmented_tweets.append(aug.augment(tweet))
        return augmented_tweets
    def augmentation_spelling(self, tweets, aug_p = 0.2, num_thread = 8):
        aug = naw.SpellingAug(aug_p = aug_p)
        augmented_tweets = aug.augment(tweets, num_thread = num_thread)
        return augmented_tweets