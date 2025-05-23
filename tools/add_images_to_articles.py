from zeeguu.core.model.article import Article
from zeeguu.core.model.url import Url
import newspaper
from zeeguu.api.app import create_app
import zeeguu.core

from time import time
from tqdm import tqdm

"""
    Goes through all articles that don't have an image to attempt to fetch an image.
    This is not necessary, I (TR) used this to test formatting on some older documents
    when searching the DB.
"""

app = create_app()
app.app_context().push()
db_session = zeeguu.core.model.db.session


def add_image_to_articles():
    all_articles = db_session.query(Article.id).filter(Article.img_url_id == None).all()
    print(f"Found {len(all_articles)} without an image.")
    total_added = 0
    for row_id in tqdm(all_articles, total=len(all_articles)):
        a_id = row_id[0]
        a = Article.find_by_id(a_id)
        if a is None:
            continue
        article_has_image = a.img_url_id is not None
        article_has_url = a.url is not None
        if article_has_image or not article_has_url:
            continue
        a_url = a.url.as_string()
        try:
            np_article = newspaper.Article(url=a_url)
            np_article.download()
            np_article.parse()
            if np_article.top_image != "":
                a.img_url = Url.find_or_create(db_session, np_article.top_image)
                total_added += 1
            if total_added >= 100:
                print("Commiting to the DB...")
                total_added = 0
                db_session.commit()
        except Exception as e:
            print(f"Failed for id: {a_id}, with: '{e}'")


def main():
    start = time()
    add_image_to_articles()
    end = time()
    print(f"Process took: {end-start}")


main()
