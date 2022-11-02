import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import IncrementalPCA
from polyglot.text import Text
from stopwordsiso import stopwords
import unicodedata
import sys

from data_utils.news_downloader import NewsDownloader
from data_utils.text_translator import TextTranslator
from data_utils.news_clusters import NewsCluster

class NewsExplorer:
    def __init__(self, source_lang="en"):
        self.source_lang = source_lang
        self.text_translator = TextTranslator()

        self.downloader = NewsDownloader(self.text_translator, source_lang=source_lang)
        
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.embedding_model = SentenceTransformer('LaBSE', device=self.device, cache_folder="/mnt/sentence_transformers_models")
        
        self.news_df = None
        self.news_embeddings = None
        self.news_clusterings = None
    
    def __get_punctuation_set(self):
        punct = set(chr(i) for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))
        return punct.union(set("1234567890"))
    
    def __embed_text(self, text_list):
        full_embeddings = self.embedding_model.encode(text_list, show_progress_bar = True)
        
        # We perform PCA to make the embeddings more relevant to the stories searched
        ipca_transformer = IncrementalPCA(n_components=None)
        return ipca_transformer.fit_transform(full_embeddings)
        
    def download_news(self, keyword):
        print("Getting news...")
        full_news_df = self.downloader.get_news(keyword)
        
        # Remove the last hyphenated piece of text from title, it is just the newspaper name - not semantically relevant
        full_news_df["content_title"] = full_news_df["title"].str.split(" - ").str[:-1].str.join(" - ")
        
        # De-duplicate the same stories (e.g. if it is on the UK and US news pages)
        groupby_df = full_news_df.groupby(["title", "source"])
        self.news_df = groupby_df[["pubDate", "description", "day_str", "language", "link", "content_title"]].first()
        self.news_df["countries"] = groupby_df["country"].apply(set).apply(list)
        
        self.news_df = self.news_df.reset_index(drop=False)
        
        # Translate titles in other languages to source language (interface language)
        for article_language in self.news_df["language"].unique():
            if article_language == self.source_lang:
                continue
            
            lang_mask = self.news_df["language"] == article_language
            
            orig_titles = self.news_df.loc[lang_mask, "content_title"].tolist()
            print(f"Translating {article_language} articles")
            translated_titles = self.text_translator.translate_text(orig_titles, article_language, self.source_lang, show_progress_bar=True)
            
            self.news_df.loc[lang_mask, "content_title"] = translated_titles
        
        # Get embeddings of the ORIGINAL (untranslated) titles
        self.news_embeddings = self.__embed_text(self.news_df.content_title.tolist())
        
        # Get words from titles
        self.news_df["words"] = self.news_df.content_title.str.lower()
        self.news_df["words"] = self.news_df["words"].apply(lambda x: Text(x, hint_language_code=self.source_lang))
        self.news_df["words"] = self.news_df["words"].apply(lambda x: x.words)
        
        # Remove stopwords from titles
        stopwords_set = stopwords(self.source_lang)
        stopwords_set.update(self.__get_punctuation_set())
        self.news_df["words"] = self.news_df["words"].apply(lambda x: [w for w in x if not w in stopwords_set])

    def get_news_clusters(self, num_topics=5):
        self.news_clusterings = NewsCluster(self.news_df, self.news_embeddings, num_topics)
        return self.news_clusterings