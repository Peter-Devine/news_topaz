from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import math
from datetime import datetime
import numpy as np
import pandas as pd
import pytz
from IPython.core.display import display, HTML
from transformers import pipeline

class NewsCluster:
    def __init__(self, news_df, news_emb, num_topics = 5, num_char_headlines = 3, num_char_keywords = 6):
        assert num_topics > 1

        self.num_char_keywords = num_char_keywords
        self.num_char_headlines = num_char_headlines
        self.max_corr = 0.2
                
        self.news_df = news_df
        self.news_emb = news_emb
        
        # Cluster embeddings to get cluster IDs
        self.news_cluster = KMeans(n_clusters=num_topics, random_state=123).fit(news_emb)
        self.news_df["cluster_ids"] = self.news_cluster.predict(news_emb)
        self.unique_cluster_ids = self.news_df["cluster_ids"].unique()
        
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
        
        # Get stats for the whole cluster
        self.entire_age = self.__get_cluster_mean_age(None)
        self.entire_size = self.__get_cluster_size(None)
        self.entire_stats = self.__get_cluster_stats(None)
        
        # Initialize zero shot classifier for 'on the fly' classification
        self.zs_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        
        # English translations of language and country codes (TODO: CHANGE THIS TO BE MULTILINGUAL EVENTUALLY!)
        self.lang_to_eng_name_dict = {'ar': 'Arabic', 'bn': 'Bengali', 'bg': 'Bulgarian', 'cs': 'Czech', 'de': 'German', 'el': 'Greek', 'en': 'English', 'fr': 'French', 'he': 'Hebrew', 'hi': 'Hindi', 'hu': 'Hungarian', 'id': 'Indonesian', 'it': 'Italian', 'ja': 'Japanese', 'ko': 'Korean', 'lv': 'Latvian', 'lt': 'Lithuanian', 'ml': 'Malayalam', 'mr': 'Marathi', 'nl': 'Dutch', 'no': 'Norwegian', 'pl': 'Polish', 'pt': 'Portuguese', 'ro': 'Romanian', 'ru': 'Russian', 'sk': 'Slovak', 'sl': 'Slovenian', 'es': 'Spanish', 'sr': 'Serbian', 'sv': 'Swedish', 'ta': 'Tamil', 'te': 'Telugu', 'th': 'Thai', 'tr': 'Turkish', 'uk': 'Ukrainian', 'vi': 'Vietnamese', 'zh': 'Chinese'}
        self.country_to_eng_name_dict = {'AE': 'UAE', 'AR': 'Argentina', 'AT': 'Austria', 'AU': 'Australia', 'BD': 'Bangladesh', 'BE': 'Belgium', 'BG': 'Bulgaria', 'BR': 'Brazil', 'BW': 'Botswana', 'CA': 'Canada', 'CH': 'Switzerland', 'CL': 'Chile', 'CN': 'China', 'CO': 'Colombia', 'CU': 'Cuba', 'CZ': 'Czech Republic', 'DE': 'Germany', 'EG': 'Egypt', 'ES': 'Spain', 'ET': 'Ethiopia', 'FR': 'France', 'GB': 'UK', 'GH': 'Ghana', 'GR': 'Greece', 'HK': 'Hong Kong', 'HU': 'Hungary', 'ID': 'Indonesia', 'IE': 'Ireland', 'IL': 'Israel', 'IN': 'India', 'IT': 'Italy', 'JP': 'Japan', 'KE': 'Kenya', 'KR': 'South Korea', 'LB': 'Lebanon', 'LT': 'Lithuania', 'LV': 'Latvia', 'MA': 'Morocco', 'MX': 'Mexico', 'MY': 'Malaysia', 'NA': 'Namibia', 'NG': 'Nigeria', 'NL': 'Netherlands', 'NO': 'Norway', 'NZ': 'New Zealand', 'PE': 'Peru', 'PH': 'Philippines', 'PK': 'Pakistan', 'PL': 'Poland', 'PT': 'Portugal', 'RO': 'Romania', 'RS': 'Serbia', 'RU': 'Russia', 'SA': 'Saudi Arabia', 'SE': 'Sweden', 'SG': 'Singapore', 'SI': 'Slovenia', 'SK': 'Slovakia', 'SN': 'Senegal', 'TH': 'Thailand', 'TR': 'Turkey', 'TW': 'Taiwan', 'TZ': 'Tanzania', 'UA': 'Ukraine', 'UG': 'Uganda', 'US': 'USA', 'VE': 'Venezuela', 'VN': 'Viet Nam', 'ZA': 'South Africa', 'ZW': 'Zimbabwe'}

        
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
        top_keywords = sorted_words[-self.num_char_keywords:]
        # Sort them backwards so that top is first, not last, in list
        top_keywords = top_keywords[::-1]
        
        return top_keywords
    
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
        
    def __get_list_series_norm_freq(self, list_series):
        # Gets the relative frequency of values in a series of lists - NB it calculates per LIST not per exploded item.
        mlb = MultiLabelBinarizer(sparse_output=True)

        one_hot_df = pd.DataFrame.sparse.from_spmatrix(
                        mlb.fit_transform(list_series),
                        index=list_series.index,
                        columns=mlb.classes_)

        return one_hot_df.mean().sort_values(ascending=False)
        
    def __get_keyword_stats(self, selected_news_df):
        zs_cols = [c for c in selected_news_df.columns if "_zs_cls" in c]
        
        keyword_stats_dict = {}
        
        for zs_col in zs_cols:
            keyword_name = zs_col[:-len("_zs_cls")]
            keyword_stats_dict[keyword_name] = selected_news_df[zs_col].mean()
        return keyword_stats_dict
        
    def __get_cluster_stats(self, cluster_id, stat_cols=["language", "source"], explode_cols=["countries"], n_stats=3):
        if cluster_id is None:
            selected_news_df = self.news_df
        else:
            clust_mask = self.news_df["cluster_ids"] == cluster_id
            selected_news_df = self.news_df[clust_mask]
        
        cluster_stats = {}
        
        for stat_col in stat_cols:
            cluster_stats[stat_col] = selected_news_df[stat_col].value_counts(normalize=True)[:n_stats].to_dict()
            
        for stat_col in explode_cols:
            cluster_stats[stat_col] = self.__get_list_series_norm_freq(selected_news_df[stat_col])[:n_stats].to_dict()
            
        cluster_stats["keywords"] = self.__get_keyword_stats(selected_news_df)
            
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
        
    def __make_link(self, link, prefix, key):
        return display(HTML(f"{prefix} <a href={link}>{key}</a>"))    

    def print_article_list(self, article_list):

        for article in article_list:
            date_str = article["day_str"]
            source_str = article["source"]
            countries_str = ", ".join([self.country_to_eng_name_dict[x] for x in article["countries"]])
            language_str = "[" + self.lang_to_eng_name_dict[article["language"]] + "] "
            translated_headline = article["content_title"]
            link = article["link"]

            self.__make_link(link, language_str, translated_headline)
            print(source_str + "      " + date_str + "      " + countries_str)
            print()
    
    def print_stats(self, stats):
        print("Top languages:")
        for lang, freq in stats["language"].items():
            print(f"{self.lang_to_eng_name_dict[lang]} - {freq * 100:.2f}%")
        print()

        print("Top countries:")
        for country, freq in stats["countries"].items():
            print(f"{self.country_to_eng_name_dict[country]} - {freq * 100:.2f}%")
        print()

        print("Top sources:")
        for source, freq in stats["source"].items():
            print(f"{source} - {freq * 100:.2f}%")
        print()
        
        if len(stats["keywords"].keys()) > 0:
            print("Search term avg. score:")
            for keyword, score in stats["keywords"].items():
                print(f"{keyword} - {score * 100:.2f}%")
            print()

    def print_cluster_info(self, cluster_info):
        
        cluster_id = cluster_info["id"]
        print(f"Cluster {cluster_id}")
        print("Headlines:")
        print("\n".join(["* " + x for x in cluster_info["headlines"]]))
        print()
        print("Average age of story:")
        print(cluster_info["age"])
        print()
        print("Number of stories:")
        print(cluster_info["size"])
        print()
        print("Keywords:")
        print(", ".join(cluster_info["keywords"]))
        print()
        self.print_stats(cluster_info["stats"])
        print()
        
    def print_all_articles(self):
        self.print_article_list(self.news_df.to_dict(orient="records"))
        
    def print_all_stats(self):
        self.print_stats(self.entire_stats)
    
    def clear_keyword_search(self):
        self.news_df = self.news_df.drop([c for c in self.news_df.columns if "_zs_cls" in c], axis=1)
    
    def search_for_keywords(self, keywords):
        zs_scores = self.zs_classifier(self.news_df.content_title.tolist(), keywords)
        
        zs_score_df = pd.DataFrame([{l: s for l, s in zip(item["labels"], item["scores"])} for item in zs_scores], index=self.news_df.index)
        
        zs_score_df.columns = [f"{c}_zs_cls" for c in zs_score_df.columns]
        
        self.clear_keyword_search()
        
        self.news_df = self.news_df.join(zs_score_df)
        
        
    def print_all_cluster_info(self):
        all_cluster_info = self.get_cluster_info()
        for cluster_id, cluster_info in all_cluster_info.items():
            self.print_cluster_info(cluster_info)
            print()
            print()
                
    def get_subclusters(self, cluster_id, num_topics=5):
        
        cluster_mask = self.news_df["cluster_ids"] == cluster_id
        
        return NewsCluster(self.news_df[cluster_mask], self.news_emb[cluster_mask], num_topics)