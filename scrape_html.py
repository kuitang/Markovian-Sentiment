from models import Blog, load

from lxml.html import parse
import sys, os, re, argparse, cPickle, string

stripwords_re = re.compile(r'http://[^\s]*|via', flags=re.IGNORECASE)

parser = argparse.ArgumentParser(description='Scrape dumped HTML into structured document features.')
parser.add_argument('blogdir', nargs='+')

def extract_posts(dom):
    posts = dom.xpath("//*[contains(@class, 'post') or contains(@class, 'posts') or contains(@id, 'post') or contains(@id, 'posts') or contains(@class, 'post-panel') or contains(@class, 'panel')]")
    articles = dom.xpath("//article")
    return posts + articles

def text_from_post(post_dom):
    # Blacklist step: remove junk from post_dom
    blacklist_xpaths = (
            "//*[contains(@class, 'meta') or contains(@id, 'meta')]",
            "//*[contains(@class, 'tags') or contains(@id, 'tags')]",
            "//a",
            "//script"
        )
    for xp in blacklist_xpaths:
        for dom in post_dom.xpath(xp):
            dom.clear()

#    # Whitelist step: assume text only comes from paragraphs
#    text = ''
#    for p_dom in post_dom.findall("p"):
#        text += p_dom.text_content()
#
    text = re.sub(stripwords_re, '', post_dom.text_content())
    return text

def make_dataset(blogdir):
    # Find the largest number here
    blog = Blog()
    entries = [ int(f) for f in os.listdir(blogdir) if f.isdigit() ]
    entries.sort()
    print "Blog %s has %d pages"%(blogdir, entries[-1])

    for page in entries:
        doc_dom = parse(os.path.join(blogdir, str(page)))
        posts = extract_posts(doc_dom)
        print "Page %d/%d had %d posts"%(page, entries[-1], len(posts))
        for post_dom in posts:
            blog.add_doc(text_from_post(post_dom))
    
#    blog.vectorize()
    # Save it
    print "Saving preprocessing..."
    cPickle.dump(open(os.path('data', os.path.basename(blogdir) + '.cache'), 'w'))
    return blog

if __name__ == "__main__":
    print "Main longer implemented."
#    args = parser.parse_args()
#    newdir = os.path.join('data', 'blogs')
#    # Load the models files
#    load()
#    if not os.path.exists(newdir): os.mkdir(newdir)
#
#    for dir in args.blogdir:
#        print "Reading %s"%dir
#        docs = make_dataset(dir)
#        name = os.path.basename(dir)
#        print "Writing %s"%name
#        with open(os.path.join(newdir, name), 'w') as f:
#            cPickle.dump(docs, f, -1)
#
