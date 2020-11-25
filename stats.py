
def articles_stats(histories):
    total_articles = 0
    total_articles_text = 0

    a = []
    b = []
    for history in histories:
        for group in history.groups:

            group_articles = len(group.search['articles']) 
            total_articles += group_articles

            if group_articles == 0:
                a.append(group)
            else:
                group_articles_text = len([article for article in group.search['articles'] if 'text' in article and len(article['text']) > 0])
                total_articles_text += group_articles_text
                if group_articles_text == 0:
                    b.append(group)

    print(f'Total number of articles: {total_articles}')
    print(f'Total number of articles with text content: {total_articles_text}')


