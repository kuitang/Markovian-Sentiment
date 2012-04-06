from models import Blog

from lxml.html import parse
from unidecode import unidecode
import sys, os, re, argparse, cPickle, string
url_re = re.compile(r'http://[^\s]*')

parser = argparse.ArgumentParser(description='Scrape dumped HTML into structured document features.')
parser.add_argument('blogdir', nargs='+')

def extract_posts(dom):
    return dom.findall("//div[@id='posts']")

def text_from_post(post_dom):
    # Remove junk from post_dom
    post_dom.remove(post_dom.find("div[@id='info']"))

    for a_dom in post_dom.findall("a"): post_dom.remove(a)

    script_dom = post_dom.find("script")
    if script_dom: post_dom.remove(script_dom)

    return re.sub(url_re, '', unidecode(post_dom.text_content()))

def make_dataset(blogdir):
    # Find the largest number here
    blog = Blog()
    entries = [ int(f) for f in os.listdir(blogdir) if f.isdigit() ]
    entries.sort()

    for page in entries:
        doc_dom = parse(os.path.join(blogdir, str(page)))
        for post_dom in extract_posts(doc_dom):
            blog.add_doc(text_from_post(post_dom))
    
    blog.freeze()
    return blog

if __name__ == "__main__":
    args = parser.parse_args()
    newdir = os.path.join('data', 'blogs')
    if not os.path.exists(newdir): os.mkdir(newdir)

    for dir in args.blogdir:
        print "Reading %s"%dir
        docs = make_dataset(dir)
        name = os.path.basename(dir[:-1])
        print "Writing %s"%name
        with open(os.path.join(newdir, name), 'w') as f:
            cPickle.dump(docs, f, -1)

