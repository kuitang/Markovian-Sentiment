import models
import numpy as np
import matplotlib.pyplot as plt
import argparse
import scrape_html
import inference


parser = argparse.ArgumentParser(description='Driver program for sentiment analysis')
parser.add_argument('blogdir')

if __name__ == "__main__":
    models.load()
    args = parser.parse_args()
    blog = scrape_html.make_dataset(args.blogdir)
    # Diagnostic plots
#    plt.hist(blog.subj_assign, len(models.SUBJECTIVITIES))
#    plt.show()
#    plt.hist(blog.sent_assign, len(models.SENTIMENTS))
#    plt.show()

    pi, theta, phi = inference.train_subjlda(blog)
#    plt.hist(blog.subj_assign, len(models.SUBJECTIVITIES))
#    plt.show()
#    plt.hist(blog.sent_assign, len(models.SENTIMENTS))
#    plt.show()
    # Print most probable words
    Njr = blog.counts[4]
    idxs = np.argsort(Njr, 1)
    for e, irow in zip(models.SENTIMENTS, idxs):
        print e, [ blog.lexicon.words[i] for i in irow[::-1][:10] ]
    
