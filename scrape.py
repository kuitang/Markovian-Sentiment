from restipy import restipy
import secrets

from pprint import pprint
import argparse, sys, time

parser = argparse.ArgumentParser(description='Scrape some users blogs.')
parser.add_argument('blog', nargs='+')

def is_success(response):
    if response['meta']['status'] != 200:
        print "Error:"
        pprint(response['meta'])
        return False
    return True

def set_title_and_body(p):
    title, body = '', ''
    titles = ('title', 'caption')
    bodies = ('body')
    for t in titles:
        if t in p:
            title = p[t]

    for b in bodies:
        if b in p:
            body = p[b]

    return title, body

def scrape(base_hostname):
    tumblr_request = restipy.make_requestor(lambda method, args_str:
        'http://api.tumblr.com/v2/blog/%s/%s?%s&api_key=%s'%(
        base_hostname, method, args_str, secrets.TUMBLR_SECRET_KEY), debug=True)

    blog_info = tumblr_request('info')
    if not is_success(blog_info): return

    pprint(blog_info['response'])

    offset = 0
    while True:
        # avoid ratelimiting
        time.sleep(1)
        
        blog = tumblr_request('posts', offset=offset, filter='text')
        if not is_success(blog): break

        # TODO: check if we should + 1.
        nposts = len(blog['response']['posts'])
        print nposts, blog['response']['total_posts']
        #assert(nposts == int(blog['response']['total_posts']))
        offset += nposts
        for p in blog['response']['posts']:
            t = p['type']

            title, body = set_title_and_body(p)
            print t, title, len(body), body

if __name__ == "__main__":
    args = parser.parse_args()
    for blog in args.blog: scrape(blog)



