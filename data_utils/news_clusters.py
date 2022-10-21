from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.feature_extraction.text import TfidfVectorizer
import math
from datetime import datetime
import numpy as np

class NewsCluster:
    def __init__(self, news_df, news_emb, num_topics = 5, num_char_headlines = 2, num_char_keywords = 6):
        assert num_topics > 1
        
        self.news_df = news_df
        self.news_emb = news_emb
        
        # Cluster embeddings to get cluster IDs
        self.news_cluster = KMeans(n_clusters=num_topics, random_state=123).fit(news_emb)
        self.news_df["cluster_ids"] = self.news_cluster.predict(news_emb)
        
        # Get TF-IDF weights for this cluster
        vectorizer = TfidfVectorizer(    
            analyzer='word',
            tokenizer=lambda x: x,
            preprocessor=lambda x: x,
            binary=True,
            min_df=3,
        )
        self.tfidf_values = vectorizer.fit_transform(df["words"].tolist())
        self.tfidf_words = vectorizer.get_feature_names_out()
        self.num_char_keywords = num_char_keywords

        self.num_char_headlines = num_char_headlines
    
    
    def __get_characteristic_headlines(self, cluster_id):
        
        clust_mask = self.news_df["cluster_ids"] == cluster_id

        distances = cdist([self.news_cluster.cluster_centers_[cluster_id]], self.news_emb[clust_mask], metric="cosine")[0]

        sorted_headline_to_centroid = self.news_df.title.iloc[distances.argsort()].tolist()

        # Take the top third of the characteristic headlines
        selected_len = int(len(sorted_headline_to_centroid) / 3)
        selected_headlines = sorted_headline_to_centroid[:selected_len]

        sample_idx = [int(selected_len * (i / self.num_char_headlines)) for i in range(self.num_char_headlines)]

        chosen_headlines = [selected_headlines[idx] for idx in sample_idx]
                        
        return chosen_headlines
       
    def __get_cluster_keywords(self, cluster_id):
        cluster_ids = self.__get_cluster_ids()
        
        clust_mask = self.news_df["cluster_ids"] == cluster_id

        sorted_args = self.tfidf_values[clust_mask].sum(axis=0).argsort()
        sorted_words = self.tfidf_words[sorted_args]
        
        # Get top (i.e. highest tf-idf score) keywords for this cluster
        raise sorted_words[-self.num_char_keywords:]
    
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
        clust_mask = self.news_df["cluster_ids"] == cluster_id
        
        dates = pd.to_datetime(self.news_df.loc[clust_mask, "pubDate"])        
        age_hours = (datetime.now() - dates) / np.timedelta64(1, 'h')
        
        average_hours = age_hours.mean()
        
        return self.__parse_age_to_readable(average_hours)
    
    def __get_cluster_size(self, cluster_id):
        clust_mask = self.news_df["cluster_ids"] == cluster_id
        return self.__parse_num_to_readable(clust_mask.sum())
        
    def __get_cluster_stats(self, cluster_id):
        raise Exception("Not implemented cluster keyword extraction")
        
    def __get_all_cluster_news(self, cluster_id):
        raise Exception("Not implemented cluster keyword extraction")

    def get_cluster_info(self):
        
        cluster_info = {}
        
        for cluster_id in self.unique_cluster_ids:
            cluster_info["id"] = cluster_id
            cluster_info["headlines"] = self.__get_characteristic_headlines(cluster_id)
            cluster_info["keywords"] = self.__get_cluster_keywords(cluster_id)
            cluster_info["age"] = self.__get_cluster_mean_age(cluster_id)
            cluster_info["size"] = self.__get_cluster_size(cluster_id)
            cluster_info["stats"] = self.__get_cluster_stats(cluster_id)
            
            cluster_info["all_news"] = self.__get_all_cluster_news(cluster_id)

            
    
#     def get_cluster_sample(self, cluster_id=None, num_sampled=5):
        
#         cluster_ids = self.__get_cluster_ids(cluster_id)
                
#         char_headlines_dict = {}
        
#         for individual_cluster_ids in cluster_ids:
#             cluster_mask = self.news_df["cluster_ids"] == individual_cluster_ids
            
#             if cluster_mask.sum() < 1:
#                 continue
            
#             char_headlines_dict[individual_cluster_ids] = self.news_df.loc[cluster_mask].sample(n = min(cluster_mask.sum(), num_sampled))
        
#         return char_headlines_dict

    
    def get_subclusters(self, cluster_id, num_topics=5):
        
        cluster_mask = self.news_df["cluster_ids"] == cluster_id
        
        return NewsCluster(self.news_df[cluster_mask], self.news_emb[cluster_mask], num_topics)

    
    