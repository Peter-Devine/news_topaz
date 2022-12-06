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
    def __init__(self, source_lang=None):
        
        if source_lang is None:
            source_lang = self.__select_language()

        self.source_lang = source_lang
        self.text_translator = TextTranslator()

        self.downloader = NewsDownloader(self.text_translator, source_lang=source_lang)
        
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.embedding_model = SentenceTransformer('LaBSE', device=self.device, cache_folder="/mnt/sentence_transformers_models")
        
        self.news_df = None
        self.news_embeddings = None
        self.news_clusterings = None
        
        self.pca_dims = 50
            
    # Input the language for searching the news
    def __select_language(self):
        lang_to_name_dict = {'sv': 'svenska', 'ml': 'മലയാളം', 'ja': '日本語', 'uk': 'Українська мова', 'de': 'Deutsch', 'lv': 'Latviešu valoda', 'bn': 'বাংলা', 'hu': 'magyar nyelv', 'sr': 'српски', 'zh': '中文', 'it': 'italiano', 'sl': 'slovenski jezik', 'fr': 'français', 'cs': 'čeština', 'pt': 'português', 'es': 'español', 'lt': 'lietuvių kalba', 'th': 'ภาษาไทย', 'hi': 'हिन्दी', 'el': 'Νέα Ελληνικά', 'he': 'עברית', 'ru': 'русский язык', 'ko': '한국어', 'ar': 'العَرَبِيَّة', 'ta': 'தமிழ்', 'id': 'bahasa Indonesia', 'sk': 'slovenčina', 'pl': 'Język polski', 'nl': 'Nederlands', 'no': 'norsk', 'tr': 'Türkçe', 'bg': 'български език', 'en': 'English', 'vi': 'Tiếng Việt', 'mr': 'मराठी', 'te': 'తెలుగు', 'ro': 'limba română'}
        
        name_to_lang_dict = {n: l for l, n in lang_to_name_dict.items()}
        lang_to_name_str = "\n".join([f"{n} ({l})" for l, n in lang_to_name_dict.items()])

        source_lang = input(f"Display language:\n{lang_to_name_str}\n").lower()

        lower_lang_names = [x.lower() for x in name_to_lang_dict.keys()]
        source_lang = name_to_lang_dict[source_lang] if source_lang in lower_lang_names else source_lang

        assert source_lang in self.lang_to_name_dict.keys(), f"Please select a supported language. {source_lang} is not supported"
        
        return source_lang
    
    def __get_punctuation_set(self):
        # Get a list of all possible punctuation in unicode so we can discard them in keywords
        punct = set(chr(i) for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))
        return punct.union(set("1234567890"))
    
    def __embed_text(self, text_list):
        full_embeddings = self.embedding_model.encode(text_list, show_progress_bar = True)
        
        # We perform PCA to make the embeddings more relevant to the stories searched
        ipca_transformer = IncrementalPCA(n_components=self.pca_dims)
        return ipca_transformer.fit_transform(full_embeddings)
        
    def __isint(self, string):
        try:
            int(string)
            return True
        except:
            return False
        
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
        self.news_df["words"] = self.news_df["words"].apply(lambda x: [w for w in x if (not w in stopwords_set) and (len(w) > 1)])

    def get_news_clusters(self, num_topics=5):
        self.news_clusterings = NewsCluster(self.news_df, self.news_embeddings, num_topics)
        return self.news_clusterings

    def __take_command(self):
        command_list = ["print_info",
                "print_articles",
                "print_stats",
                "keyword_search",
                "remove_keywords",
                "subcluster",
                "reset",
                "export",
                "quit"]

        command_list_str = ",  ".join([f"{i}. {c}" for i, c in enumerate(command_list)])

        command_str = input(f"Choose one of the following commands:\n\n{command_list_str}")

        if self.__isint(command_str):
            command_str = command_list[int(command_str)]

        if command_str not in command_list:
            print(f"Command ({command_str}) is not recognised!")
            return take_command()

        return command_str


    def explore(self):
        # Interactive input based exploration of the clustered news, allowing for subclustering, printing of cluster summary, all cluster articles, all cluster stats, and searching for keywords.

        command = self.__take_command()

        current_clustering = self.news_clusterings

        while command != "quit":
            if command == "print_info":
                current_clustering.print_all_cluster_info()
            if command == "print_articles":
                current_clustering.print_all_articles()
            if command == "print_stats":
                current_clustering.print_all_stats()
            if command == "keyword_search":
                search_term = input("Please enter your search terms (comma separated if more than one): ")
                search_terms = search_term.split(",")
                current_clustering.search_for_keywords(search_terms)
            if command == "remove_keywords":
                current_clustering.clear_keyword_search()
            if command == "subcluster":
                cluster_id = input("Please enter ID of cluster to subcluster:")

                while not self.__isint(cluster_id):
                    cluster_id = input("Please enter ID of cluster to subcluster:")

                cluster_id = int(cluster_id)
                subcluster = current_clustering.get_subclusters(cluster_id)
                current_clustering = subcluster
            if command == "reset":
                current_clustering = self.news_clusterings
            if command == "export":
                current_clustering.export_cluster()

            command = self.__take_command()
