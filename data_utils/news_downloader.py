import urllib.parse
from tqdm.notebook import tqdm
from lxml import etree
import requests
import random
import time
import pandas as pd

class NewsDownloader:
    
    def __init__(self, text_translator, source_lang="en"):
        self.source_lang = source_lang
        self.text_translator = text_translator
        
        self.supported_lang_countries = [('en', 'US'), ('ja', 'JP'), ('en', 'AU'), ('en', 'BW'), ('en', 'CA'), ('en', 'ET'), ('en', 'GH'), ('en', 'IN'), ('en', 'ID'), ('en', 'IE'), ('en', 'IL'), ('en', 'KE'), ('en', 'LV'), ('en', 'MY'), ('en', 'NA'), ('en', 'NZ'), ('en', 'NG'), ('en', 'PK'), ('en', 'PH'), ('en', 'SG'), ('en', 'ZA'), ('en', 'TZ'), ('en', 'UG'), ('en', 'GB'), ('en', 'ZW'), ('id', 'ID'), ('cs', 'CZ'), ('de', 'DE'), ('de', 'AT'), ('de', 'CH'), ('es', 'AR'), ('es', 'CL'), ('es', 'CO'), ('es', 'CU'), ('es', 'ES'), ('es', 'US'), ('es', 'MX'), ('es', 'PE'), ('es', 'VE'), ('fr', 'BE'), ('fr', 'CA'), ('fr', 'FR'), ('fr', 'MA'), ('fr', 'SN'), ('fr', 'CH'), ('it', 'IT'), ('lv', 'LV'), ('lt', 'LT'), ('hu', 'HU'), ('nl', 'BE'), ('nl', 'NL'), ('no', 'NO'), ('pl', 'PL'), ('pt', 'BR'), ('pt', 'PT'), ('ro', 'RO'), ('sk', 'SK'), ('sl', 'SI'), ('sv', 'SE'), ('vi', 'VN'), ('tr', 'TR'), ('el', 'GR'), ('bg', 'BG'), ('ru', 'RU'), ('ru', 'UA'), ('sr', 'RS'), ('uk', 'UA'), ('he', 'IL'), ('ar', 'AE'), ('ar', 'SA'), ('ar', 'LB'), ('ar', 'EG'), ('mr', 'IN'), ('hi', 'IN'), ('bn', 'BD'), ('bn', 'IN'), ('ta', 'IN'), ('te', 'IN'), ('ml', 'IN'), ('th', 'TH'), ('zh', 'CN'), ('zh', 'TW'), ('zh', 'HK'), ('ko', 'KR')]
        self.lang_set = {lang for lang, country in self.supported_lang_countries}
        self.sleep_num = 2
    
    # Get the URL params for each country/language given a keyword
    def get_urls_from_keyword(self, keyword):
    
        url_list = []

        # Telugu not supported by m2m100 right now :(
        print("Translating keywords")
        keyword_map = {lang: self.text_translator.translate_text([keyword], self.source_lang, lang)[0] for lang in tqdm(self.lang_set) if lang not in [self.source_lang, "te"]}

        print(keyword_map)

        for lang, country in self.supported_lang_countries:
            # Telugu not supported by m2m100 right now :(
            if lang == "te":
                continue

            translated_keyword = keyword if lang == self.source_lang else keyword_map[lang]

            params = {"q": translated_keyword, "hl": lang, "gl": country, "ceid": f"{country}:{lang}"}
            url_params = urllib.parse.urlencode(params)

            url_list.append((url_params, lang, country))

        return url_list
    
    # Download the news given a URL and parse into dataframe
    def get_news_df(self, url, lang, country):
        time.sleep(self.sleep_num * random.random())

        r = requests.get(url)
        news_contents = etree.XML(r.content)
        items = news_contents.xpath("//item")

        if len(items) < 1:
            return None

        headline_data = [{y.tag: y.text for y in x} for x in items]

        df = pd.DataFrame(headline_data)

        df["language"] = lang
        df["country"] = country

        df["day_str"] = pd.to_datetime(df.pubDate).dt.strftime("%Y-%m-%d-%H-%M")

        return df
    
    # Get dataframe of news given keyword
    def get_news(self, keyword):
        url_params = self.get_urls_from_keyword(keyword)

        url_root = "https://news.google.com/rss/search?"

        print("Downloading news")
        df_list = [self.get_news_df(url_root + url_param, lang, country) for url_param, lang, country in tqdm(url_params)]

        return pd.concat(df_list)
