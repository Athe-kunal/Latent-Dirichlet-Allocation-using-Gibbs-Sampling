
from algorithms import Algorithms
import time
import argparse

if __name__ == '__main__':
    # artificial_algo = Algorithms("pp4data\\artificial",K=2)

    # start_time = time.time()
    # artificial_algo.LDATopicModelling()
    # print('For LDA it took',time.time()-start_time, ' seconds')

    # artificial_algo.write_to_csv('topicwords_artificial.csv')
    # artificial_algo.write_matrix('artificial')
    # start_time = time.time()
    # bow_newsgroup_error, cd_newsgroup_error = artificial_algo.run_logistic()
    # print('For training it took',time.time()-start_time,' seconds')
    # artificial_algo.plot_error(bow_newsgroup_error, cd_newsgroup_error,'artificial')

    newsgroup_algo = Algorithms("pp4data\\20newsgroups",K=20)
    

    start_time = time.time()
    newsgroup_algo.LDATopicModelling()
    # For LDA it took 4037.7739000320435  seconds
    print('For LDA it took',time.time()-start_time,' seconds')
    newsgroup_algo.write_matrix('Newsgroup')
    
    C_d,C_t = newsgroup_algo.get_matrix('Newsgroup')
    # print(C_d,C_t)
    newsgroup_algo.write_to_csv('topicwords.csv',C_t)
    bow_newsgroup_error, cd_newsgroup_error,bow_time,cd_time = newsgroup_algo.run_logistic(C_d)
    newsgroup_algo.plot_error(bow_newsgroup_error, cd_newsgroup_error,'Newsgroup-20')

    print('BOW Model took ',bow_time,' seconds')
    print('Topic-LDA Model took ',cd_time, ' seconds')
