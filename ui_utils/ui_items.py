# from IPython.html import widgets
from IPython.display import display
from ipywidgets import GridspecLayout, HBox, VBox, Label, Button, AppLayout, Dropdown, Layout, Text, HTML, Box
import math

def get_search_bar(search_fn):
    supported_langs = ['ar', 'bg', 'bn', 'cs', 'de', 'el', 'en', 'es', 'fr', 'he', 'hi', 'hu', 'id', 'it', 'ja', 'ko', 'lt', 'lv', 'ml', 'mr', 'nl', 'no', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'sr', 'sv', 'ta', 'te', 'th', 'tr', 'uk', 'vi', 'zh']

    lang_dropdown = Dropdown(
        options=supported_langs,
        value='en',
        description='Search language',
        disabled=False,
        style={'description_width': 'initial'},
        layout=Layout(width='100%')
    )
    text = Text(layout = Layout(width='100%'), placeholder="Keyword")
    search_button = Button(description="Search", layout = Layout(width='auto'))
    
    search_button.on_click(lambda x: search_fn(text.value, lang_dropdown.value))

    search_bar_obj = AppLayout(
        left_sidebar = lang_dropdown,
        center = text, 
        right_sidebar = search_button,
        layout=Layout(width='100%', height="100%", padding="0px 0px 30px 0px")
    )
    
    return search_bar_obj

def get_subheader_value(text):
    return f"<h4>{text}</h4>"

def get_header_value(text):
    return f"<h1>{text}</h1>"

def get_info_bar(header_name, return_fn, reset_fn, cluster_gran_fn):
    return_button = Button(icon='arrow-left', layout=Layout(height="auto"))
    return_button.on_click(return_fn)

    reset_button = Button(icon='rotate-left', layout=Layout(height="auto"))
    reset_button.on_click(reset_fn)
    
    lab = HTML(value=get_subheader_value(header_name), layout=Layout(width='100%', height="100%", padding="0px 0px 0px 20px"))
    
    granularity_dropdown = Dropdown(
        options=[('Coarse', 3), ('Medium', 6), ('Fine', 9)],
        value=6,
        description='Topic granularity',
        disabled=False,
        style={'description_width': 'initial'}    
    )
    
    def on_change(change):
        if change['type'] == 'change' and change['name'] == 'value':
            cluster_gran_fn(change['new'])
            
    granularity_dropdown.observe(on_change)
    
    info_bar = HBox([return_button, reset_button, lab, granularity_dropdown])
    
    return info_bar

# info_bar.layout.visibility = 'hidden'

def search_function(keyword, language):
    print(f"{keyword}_{language}")
    
    # Set info bar text
    info_bar.children[2].value = get_subheader_value(keyword)
    
    info_bar.layout.visibility = 'visible'

def get_padded_label(name, y_padding, x_padding):
    label_layout = Layout(padding=f'{y_padding}px {x_padding}px')
    padded_label = Label(value=name, layout=label_layout)
    padded_label.add_class(f"no-margin")
    padded_label.add_class(f"stat-label")
    return padded_label

def add_theme(item, theme, outline=False):
    
    if outline:
        item.add_class(f"text-{theme}")
        item.add_class(f"border")
        item.add_class(f"border-{theme}")
        item.add_class(f"rounded")
    else:
        item.add_class(f"bg-{theme}")
    
        if theme in ["primary", "secondary", "success", "danger", "dark"]:
            item.add_class(f"text-white")
        
    return item
    
def get_stat_item(name, magnitude, age, data_type):
    
    if data_type == "language":
        theme = "success"
    elif data_type == "country":
        theme = "danger"
    elif data_type == "source":
        theme = "warning"
    else:
        theme = "light"
    
    grid = GridspecLayout(1, 18)
    
    n_label = get_padded_label(name, 0, 10)
    m_label = get_padded_label(str(magnitude), 0, 10)
    a_label = get_padded_label(str(age), 0, 10)
    
    add_theme(n_label, theme)

    n_label.add_class(f"m-0")
    n_label.add_class(f"p-0")
    n_label.add_class(f"ps-2")
    n_label.add_class(f"fw-bold")
    
    m_label.add_class(f"m-0")
    m_label.add_class(f"p-0")
    m_label.add_class(f"ps-2")
    
    a_label.add_class(f"bg-dark")
    a_label.add_class(f"text-white")
    a_label.add_class(f"m-0")
    a_label.add_class(f"p-0")
    a_label.add_class(f"ps-2")
    
    grid[0, :8] = n_label
    grid[0, 8:13] = m_label
    grid[0, 13:] = a_label
    
    grid.add_class("border")
    grid.add_class("border-2")
    grid.add_class("border-secondary")
    grid.add_class("rounded")
    grid.add_class("mb-1")
    grid.add_class("mt-1")

    return grid


def get_link(text, link):
    return HTML(value=f"<a href='{link}'>{text}</a>")

def get_news_stat(text, theme):
    news_stat = get_minitag(text, theme=theme)
    news_stat.add_class("px-2")
    return news_stat

def get_news_item(headline, link, language, country, source, date):    
    headline_text = get_link(get_header_text(headline, header_depth=4), link)
    headline_text.add_class("my-3")
    
    lang_stat = HTML(value=get_header_text(language, header_depth=4))
    lang_stat.add_class("rounded")
    lang_stat.add_class("rounded-3")
    lang_stat.add_class("my-0")
    lang_stat.add_class("text-center")
    add_theme(lang_stat, "light")
    lang_stat.add_class("py-1")
    lang_stat.add_class("px-2")
    lang_stat.add_class("border")
    lang_stat.add_class("border-2")
    lang_stat.add_class("border-secondary")

    
    headline_data = HBox([
        lang_stat,
        headline_text
    ])
    
    
    date_text = HTML(value=date)
    date_text.add_class("text-secondary")
    date_text.add_class("float-end")
    
    headline_stats = HBox([
        get_news_stat(country, theme="secondary"),
        get_news_stat(source, theme="info")
    ])
    
    news_box = VBox([headline_data, date_text, headline_stats], layout=Layout(min_height='105px'))
    
    news_box.add_class("border")
    news_box.add_class("border-1")
    news_box.add_class("rounded-3")
    news_box.add_class("px-4")
    news_box.add_class("py-2")
    news_box.add_class("mb-2")
    news_box.add_class("mx-3")
    
    return news_box
    
def get_article_list(news_item_list):
    
    news_items = [get_news_item(x["headline"],
                   x["link"],
                   x["language"],
                   x["country"],
                   x["source"],
                   x["date"])
                   for x in news_item_list]
        
    news_list = VBox(news_items, layout=Layout(width='100%', max_height='500px'))

    news_list_header = get_header("Article list", header_depth=1, center = False)
    news_list_header.add_class("px-4")
    news_list_box = VBox([news_list_header, news_list])

    news_list_box.add_class("border")
    news_list_box.add_class("border-5")
    news_list_box.add_class("rounded-3")
    news_list_box.add_class("py-2")
    news_list_box.add_class("px-5")

    return news_list_box

def get_header_text(text, header_depth):
    style = """'padding: 0px; margin: 0px;'"""
    return f"<h{header_depth} style={style}>{text}</h{header_depth}>"

def get_header(text, header_depth=3, center = False):
    header = HTML(value=get_header_text(text, header_depth))
    
    if center:
        header.add_class("text-center")
    
    return header

def get_minitag(text, theme):
    minitag = HTML(value=get_header_text(text, header_depth=4))
    
    minitag.add_class("rounded")
    minitag.add_class("rounded-3")
    minitag.add_class("my-2")
    minitag.add_class("text-center")
    add_theme(minitag, theme)
    
    if theme == "light":
        minitag.add_class("py-1")
        minitag.add_class("border")
        minitag.add_class("border-2")
        minitag.add_class("border-secondary")
    else:
        minitag.add_class("py-2")    

    return minitag

def cluster_info(cluster_id,
                 cluster_title, 
                 cluster_mag, 
                 cluster_date, 
                 cluster_texts, 
                 cluster_keywords, 
                 cluster_stats,
                 expand_fn,
                 subcluster_fn):
    
    
    # Header
    header = GridspecLayout(1, 12)
    cluster_title = get_header(cluster_title)
    cluster_title.add_class("ms-3")
    cluster_title.add_class("my-1")
    cluster_title.add_class("py-1")
    header[0, 0:6] = cluster_title
    header[0, 8:10] = get_minitag(cluster_mag, "info")
    header[0, 10:12] = get_minitag(cluster_date, "secondary")
    
    header.add_class("border")
    header.add_class("border-5")
    header.add_class("border-start-0")
    header.add_class("border-end-0")
    header.add_class("border-top-0")
    
    # Contents
    
    ## Headlines
    headline_objects = [get_header(t, header_depth=4) for t in cluster_texts]
    [h.add_class("m-2") for h in headline_objects]
    [h.add_class("my-3") for h in headline_objects]
    headlines_box = VBox(headline_objects)
    
    ## Char words
    char_word_objects = [get_minitag(w, theme="dark") for w in cluster_keywords]
    [cw.add_class("px-1") for cw in char_word_objects]
    char_words_box = Box(char_word_objects)
    char_words_box.add_class("overflow-hidden")
    
    ## Stats
    stat_tags_grid = GridspecLayout(1, 2)
    
    stat_tags = [get_stat_item(cs["text"], cs["mag"], cs["age"], cs["data_type"]) for cs in cluster_stats] * 6
    
    stat_boxes1 = VBox(stat_tags[0:3])
    stat_boxes1.add_class("px-1")
    stat_tags_grid[0, 0] = stat_boxes1
    
    stat_boxes2 = VBox(stat_tags[3:6])
    stat_boxes2.add_class("px-1")
    stat_tags_grid[0, 1] = stat_boxes2
    
    contents = VBox([headlines_box, char_words_box, stat_tags_grid])
    
    # Footer
    footer = GridspecLayout(1, 12)
    plus_button = Button(icon='plus')
    plus_button.on_click(lambda x: expand_fn(cluster_id))
    footer[0, 8:10] = plus_button
    
    cluster_button = Button(icon='object-group')
    cluster_button.on_click(lambda x: subcluster_fn(cluster_id))
    footer[0, 10:] = cluster_button
    
    footer.add_class("mt-2")
    footer.add_class("border")
    footer.add_class("border-5")
    footer.add_class("border-start-0")
    footer.add_class("border-bottom-0")
    footer.add_class("border-end-0")
    
    cluster_v_box = VBox([header, contents, footer])
        
    cluster_v_box.add_class("border")
    cluster_v_box.add_class("border-4")
    cluster_v_box.add_class("rounded-3")
    cluster_v_box.add_class("mx-2")

    return cluster_v_box

def clusters_info(cluster_info_list):
    
    vertical_boxes = []
    
    n_cells_per_row = 3
    n_rows = math.ceil(len(cluster_info_list) / n_cells_per_row)
        
    for n_row in range(n_rows):
        
        start_i = n_row*n_cells_per_row
        
        horizontal_box_items = [cluster_info(c["id"],
                 c["title"], 
                 c["mag"], 
                 c["date"], 
                 c["texts"], 
                 c["keywords"], 
                 c["stats"],
                 c["expand_fn"],
                 c["subcluster_fn"]) for c in cluster_info_list[start_i:(start_i+n_cells_per_row)]]
        
        horizontal_box = HBox(horizontal_box_items)
        horizontal_box.add_class("mb-4")
        
        vertical_boxes.append(horizontal_box)
    
    return VBox(vertical_boxes)