from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.feature_extraction.text import CountVectorizer
import math
from datetime import datetime
import numpy as np
import pandas as pd
import pytz

class NewsCluster:
    def __init__(self, news_df, news_emb, num_topics = 5, num_char_headlines = 2, num_char_keywords = 6):
        assert num_topics > 1
        
        self.news_df = news_df
        self.news_emb = news_emb
        
        # Cluster embeddings to get cluster IDs
        self.news_cluster = KMeans(n_clusters=num_topics, random_state=123).fit(news_emb)
        self.news_df["cluster_ids"] = self.news_cluster.predict(news_emb)
        self.unique_cluster_ids = self.news_df["cluster_ids"]
        
        # Get TF-IDF weights for this cluster
        vectorizer = CountVectorizer(  
            encoding='utf-8',
            tokenizer=lambda x: x,
            preprocessor=lambda x: x,
            binary=True,
            min_df=5,
        )
        
        # Get embeddings
        self.tfidf_values = vectorizer.fit_transform(self.news_df["words"].tolist())
        self.tfidf_words = vectorizer.get_feature_names_out()
        self.tfidf_values, self.tfidf_words = self.__decorrelate_vocab(self.tfidf_values.toarray(), self.tfidf_words)
        self.num_char_keywords = num_char_keywords
        
        self.max_corr = 0.7

        self.num_char_headlines = num_char_headlines
        
        # Get stats for the whole cluster
        self.entire_age = self.__get_cluster_mean_age(None)
        self.entire_size = self.__get_cluster_size(None)
        self.entire_stats = self.__get_cluster_stats(None)
        
    def __decorrelate_vocab(self, encoding, vocab):
            
        # Get correlations between each row (i.e. each word)
        corr_mat = np.corrcoef(encoding.T)
        corr_mat = np.nan_to_num(corr_mat)

        # Make diagonals zero
        ind = np.diag_indices(corr_mat.shape[0])
        corr_mat[ind[0], ind[1]] = np.zeros(corr_mat.shape[0])

        # Order autocorrs to make most autocorrelated first in the index (We want to keep the most multicorrelated words)
        corr_mat_arg_sort = corr_mat.sum(axis=1).argsort()[::-1]
        # Re-sort autocorr matrix with the most autocorrelated words appearing later in the matrix
        corr_mat = corr_mat[corr_mat_arg_sort][:, corr_mat_arg_sort]
        # Re-sort encodings to align with new autocorr indices
        encoding = encoding[:, corr_mat_arg_sort]
        # Re-sort vocab to align with new autocorr indices
        vocab = vocab[corr_mat_arg_sort]

        # Get the X and Y coord of pairs of words with sim > sim_threshold
        Xs, Ys = np.where(corr_mat > self.max_corr)
        # Get the X coordinates of all coordinates where the Y is less than the X coord
        # This is done to only keep the second of each pair (We want to keep one word with this meaning existing at least!)
        words_idx_to_drop = Xs[Xs < Ys]
        # Get a bool mask relating to which indices are in list of words to drop
        corr_word_mask = np.isin(np.arange(encoding.shape[1]), words_idx_to_drop)
        # Apply bool mask to encodings
        decorr_encode = encoding[:, ~corr_word_mask]
        # Apply bool mask to vocab
        decorr_vocab = vocab[~corr_word_mask]

        return decorr_encode, decorr_vocab
        
    
    def __get_characteristic_headlines(self, cluster_id):
        
        if cluster_id is None:
            selected_news_df = self.news_df
            selected_emb = self.news_emb
            cluster_center = self.news_emb.mean(axis=0)
        else:
            clust_mask = self.news_df["cluster_ids"] == cluster_id

            selected_news_df = self.news_df[clust_mask]
            selected_emb = self.news_emb[clust_mask]
            cluster_center = self.news_cluster.cluster_centers_[cluster_id]

        distances = cdist([cluster_center], selected_emb, metric="cosine")[0]

        sorted_headline_to_centroid = selected_news_df.content_title.iloc[distances.argsort()].tolist()

        # Take the top third of the characteristic headlines
        selected_len = int(len(sorted_headline_to_centroid) / 3)
        selected_headlines = sorted_headline_to_centroid[:selected_len]

        sample_idx = [int(selected_len * (i / self.num_char_headlines)) for i in range(self.num_char_headlines)]

        chosen_headlines = [selected_headlines[idx] for idx in sample_idx]
                        
        return chosen_headlines
       
    def __get_cluster_keywords(self, cluster_id):
        if cluster_id is None:
            selected_news_df = self.news_df
            selected_tfidf = self.tfidf_values
        else:
            clust_mask = self.news_df["cluster_ids"] == cluster_id

            selected_news_df = self.news_df[clust_mask]
            
            selected_tfidf = self.tfidf_values[clust_mask, :] / (1 + (self.tfidf_values.sum(axis=0)))
        
        sorted_args = selected_tfidf.sum(axis=0).argsort()
        
        sorted_words = self.tfidf_words[sorted_args]
        
        # Get top (i.e. highest tf-idf score) keywords for this cluster
        return sorted_words[-self.num_char_keywords:]
    
    def __parse_num_to_readable(self, num):

        millnames = ['', 'K', 'M', 'B', 'T']
        n = float(num)
        millidx = max(0, min(len(millnames)-1, int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))

        return '{:.0f}{}'.format(n / 10**(3 * millidx), millnames[millidx])
    
    def __parse_age_to_readable(self, hours_age):
        
        hours_in_day = 24
        hours_in_week = hours_in_day * 7
        hours_in_year = hours_in_day * 365
        
        if hours_age < hours_in_day:
            return f"{int(hours_age)}h"
        elif hours_age < hours_in_week:
            return f"{int(hours_age / hours_in_day)}d"
        elif hours_age < hours_in_year:
            return f"{int(hours_age / hours_in_week)}w"
        else:
            return f"{int(hours_age / hours_in_year)}y"
    
    def __get_cluster_mean_age(self, cluster_id):
        
        if cluster_id is None:
            selected_news_df = self.news_df
        else:
            clust_mask = self.news_df["cluster_ids"] == cluster_id

            selected_news_df = self.news_df[clust_mask]
        
        dates = pd.to_datetime(selected_news_df["pubDate"])        
        age_hours = (datetime.now(pytz.utc) - dates) / np.timedelta64(1, 'h')
        
        average_hours = age_hours.mean()
        
        return self.__parse_age_to_readable(average_hours)
    
    def __get_cluster_size(self, cluster_id):
        if cluster_id is None:
            clust_size = self.news_df.shape[0]
        else:
            clust_mask = self.news_df["cluster_ids"] == cluster_id
            clust_size = clust_mask.sum()
        
        return self.__parse_num_to_readable(clust_size)
        
    def __get_cluster_stats(self, cluster_id, stat_cols=["language", "source"], explode_cols=["countries"], n_stats=2):
        if cluster_id is None:
            selected_news_df = self.news_df
        else:
            clust_mask = self.news_df["cluster_ids"] == cluster_id
            selected_news_df = self.news_df[clust_mask]
        
        cluster_stats = {}
        
        for stat_col in stat_cols:
            cluster_stats[stat_col] = selected_news_df[stat_col].value_counts(normalize=True)[:n_stats].to_dict()
            
        for stat_col in explode_cols:
            cluster_stats[stat_col] = selected_news_df[stat_col].explode().value_counts(normalize=True)[:n_stats].to_dict()
            
        return cluster_stats
        
    def __get_all_cluster_news(self, cluster_id, cols=[""]):
        if cluster_id is None:
            selected_news_df = self.news_df
        else:
            clust_mask = self.news_df["cluster_ids"] == cluster_id
            selected_news_df = self.news_df[clust_mask]
        
        return selected_news_df.to_dict(orient="records")
        
    def get_cluster_info(self):
        
        cluster_info = {}
        
        for cluster_id in self.unique_cluster_ids:
            cluster_info[cluster_id] = {}
            
            cluster_info[cluster_id]["id"] = cluster_id
            cluster_info[cluster_id]["headlines"] = self.__get_characteristic_headlines(cluster_id)
            cluster_info[cluster_id]["keywords"] = self.__get_cluster_keywords(cluster_id)
            cluster_info[cluster_id]["age"] = self.__get_cluster_mean_age(cluster_id)
            cluster_info[cluster_id]["size"] = self.__get_cluster_size(cluster_id)
            cluster_info[cluster_id]["stats"] = self.__get_cluster_stats(cluster_id)
            
            cluster_info[cluster_id]["all_news"] = self.__get_all_cluster_news(cluster_id)
    
        return cluster_info
                
    def get_subclusters(self, cluster_id, num_topics=5):
        
        cluster_mask = self.news_df["cluster_ids"] == cluster_id
        
        return NewsCluster(self.news_df[cluster_mask], self.news_emb[cluster_mask], num_topics)