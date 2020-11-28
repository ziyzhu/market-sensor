from collections import Counter

def article_df_stats(article_dfs):
    n_article = 0
    n_article_with_text = 0

    sources = list()
    n_source = 0
   
    for article_df in article_dfs:
        n_article += len(article_df)
        articles_with_text = article_df[article_df['text'].str.len() > 200]
        n_article_with_text += len(articles_with_text)
        sources += list(map(lambda d: d['href'], article_df['source'].tolist()))

    print(f'Total number of articles: {n_article}')
    print(f'Total number of articles with text: {n_article_with_text}')

    n_source = len(set(sources))
    source_count = Counter(sources)
    print(f'Total number of sources: {n_source}')
    print(f'Top 10 most common sources: {source_count.most_common(10)}')

