import cPickle
import models
import numpy as np
import matplotlib.pyplot as plt
import argparse
import scrape_html
import inference


parser = argparse.ArgumentParser(description='Driver program for sentiment analysis')
parser.add_argument('blogdir')
parser.add_argument('outfile', default='lastrun.dat')

def main(blogdir, outfile):
    models.load()
    blog = scrape_html.make_dataset(args.blogdir)
    # Diagnostic plots
    #plt.hist(blog.subj_assign, len(models.SUBJECTIVITIES))
    #plt.hist(blog.sent_assign, len(models.SENTIMENTS))

    pi, theta, phi = inference.train_subjlda(blog)

    result = { 'pi': pi, 'theta': theta, 'phi': phi, 'blog': blog }

    cPickle.dump(result, args.outfile, -1)
    
    # Print most probable words
    Njr = blog.counts[4]
    idxs = np.argsort(Njr, 1)
    phi_idxs = np.argsort(phi, 1)
    for e, irow, phi_irow in zip(models.SENTIMENTS, idxs, phi_idxs):
        print 'Njr', e, [ blog.lexicon.words[i] for i in irow[::-1][:10] ]
        print 'phi', e, [ blog.lexicon.words[i] for i in phi_irow[::-1][:10] ]

    print "Sentiment distribution over words"
    for i, e in enumerate(models.SENTIMENTS):
        plt.plot(phi[i,:])
        plt.show()

    plt.hist(blog.subj_assign, len(models.SUBJECTIVITIES))
    plt.show()
    plt.hist(blog.sent_assign, len(models.SENTIMENTS))
    plt.show()

    return 0

if __name__ == "__main__":
    args = parser.parse_args()
    sys.exit(main(args.blogdir, args.outfile))

