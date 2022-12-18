import numpy as np
import random
import os
import csv
import json
from logisticRegression import LogisticAlgorithms
import matplotlib.pyplot as plt
import time

plt.rcParams["figure.figsize"] = (20, 20)
plt.style.use("ggplot")

class Algorithms:
    def __init__(self, parent_dir, K: int, N_iters=500):
        self.K = K
        self.N_iters = N_iters
        self.word_dict = {}
        all_words = []
        self.document_indexes = []
        for files in os.listdir(parent_dir)[:-1]:
            read_files = parent_dir + '\\'+files
            with open(read_files, "r") as f:
                lines = f.readlines()
                l = lines[0].split(" ")[:-1]
            for word in l:
                all_words.append(word)
                self.document_indexes.append([word, int(files) - 1])
                if word in self.word_dict:
                    self.word_dict[word] += 1
                else:
                    self.word_dict.update({word: 0})
        #Document index has word,doc_name and random topic initially
        [elem.append(random.randint(0,K-1)) for elem in self.document_indexes]
        vocab_dict_ = dict(sorted(self.word_dict.items()))
        idx = [i for i in range(len(vocab_dict_))]
        self.vocab_dict = {elem: idx[i] for i, elem in enumerate(vocab_dict_.keys())}

        self.D = len(os.listdir(parent_dir)[:-1])
        self.V = len(self.word_dict)
        self.C_d = np.zeros((self.D, self.K))
        self.C_t = np.zeros((self.K, self.V))

        self.topic_indices = [i for i in range(self.K)]
        self.indx_to_word_dict = {v:k for k,v in self.vocab_dict.items()}

        for elem in self.document_indexes:
            word_ = elem[0]
            doc_ = elem[1]
            topic_ = elem[2]
            word_idx_ = self.vocab_dict[word_]
            self.C_d[doc_][topic_] += 1
            self.C_t[topic_][word_idx_] += 1

        with open(parent_dir+'\\'+'index.csv','r') as f:
            lines = f.readlines()


        label = [l.split("\n")[0].split(",")[1] for l in lines]
        self.label = [int(l) for l in label]
        print('Creating Bag-of-words')
        self.bag_of_words = []
        word_keys = sorted(self.word_dict.keys())
        list_dir = sorted([ int(i) for i in os.listdir(parent_dir)[:-1]])
        list_dir = [str(i) for i in list_dir]
        for files in list_dir:
            read_files = parent_dir+'\\'+files
            # print(read_files)
            with open(read_files,'r') as f:
                lines = f.readlines()
                l = lines[0].split(" ")[:-1]
            temp_word_dict = dict().fromkeys(word_keys,0)
            for word in l:
                temp_word_dict[word]+=1
            # print(temp_word_dict)
            self.bag_of_words.append(list(temp_word_dict.values()))

    def LDATopicModelling(self):
        beta = 0.01
        alpha = 5 / self.K
        #COllapsed Gibbs Sampler
        for iter in range(self.N_iters):
            random.shuffle(self.document_indexes)
            for idx, elem in enumerate(self.document_indexes):
                # Document index: word, doc_num and initially randomly generated topic
                word = elem[0]
                doc = elem[1]
                topic = elem[2]
                word_idx = self.vocab_dict[word]

                self.C_d[doc][topic] -= 1
                self.C_t[topic][word_idx] -= 1
                # print(C_t)
                p_ks = []
                for k in range(self.K):
                    c_t_sum = sum(self.C_t[k, :])
                    c_d_sum = self.C_d[:, k].sum()
                    P_k = (
                        (self.C_t[k][word_idx] + beta) / (self.V * beta + c_t_sum)
                    ) * ((self.C_d[doc][k] + alpha) / (self.K * alpha + c_d_sum))
                    assert P_k >= 0, f"The {self.C_t},{self.C_d},{P_k}"
                    p_ks.append(P_k)
                sum_pks = sum(p_ks)
                # print(p_ks)
                #Normalizing the data
                p_ks = [i / sum_pks for i in p_ks]
                assigned_topic = np.random.choice(self.topic_indices, p=p_ks)
                # print(assigned_topic)
                # self.z_n[idx] = assigned_topic
                #Assign the randomly chosen topic
                self.document_indexes[idx][2] = assigned_topic
                self.C_d[doc][assigned_topic] += 1
                self.C_t[assigned_topic][word_idx] += 1

            print(f"Completed for {iter+1}/{self.N_iters}")

    def write_to_csv(self,file_name,C_t=[]):
        if file_name == 'topicwords_artificial':
            num_words = 3
        else:
            num_words = 5

        if C_t==[]:
            model_Ct = self.C_t
        else:
            model_Ct = np.array(C_t)
        with open(file_name, 'w') as f:
            writer = csv.writer(f)
            for j in range(self.K):
                print(f'For topic {j}')
                words = [f'Topic {j}']
                #Sorting the first 5 words in 
                for i in np.argsort(model_Ct[j,:])[::-1][:num_words]:
                    print(self.indx_to_word_dict[i])
                    words.append(self.indx_to_word_dict[i])
                print('-------------------------------------------')
                writer.writerow(words)
    
    def write_matrix(self,file_name):

        with open(f"{file_name}_Cd", "w") as fp:
            json.dump(self.C_d.tolist(), fp)
        with open(f"{file_name}_Ct", "w") as fp:
            json.dump(self.C_t.tolist(), fp)
    
    def get_matrix(self,file_name):
        with open(f"{file_name}_Cd", "r") as fp:
            C_d = json.load(fp)
        with open(f"{file_name}_Ct", "r") as fp:
            C_t = json.load(fp)
        
        return C_d,C_t
    
    def run_logistic(self,C_d=[]):
        if C_d==[]:
            C_d_list = self.C_d.tolist()
        else:
            C_d_list = C_d

        bow_algo = LogisticAlgorithms(self.bag_of_words,self.label)
        Cd_algo = LogisticAlgorithms(C_d_list,self.label)

        data_fractions = [round(d, 1) for d in np.linspace(0.1, 1.0, 10)]
        a_epochs_error_blr_bow = {df: [] for df in data_fractions}
        a_epochs_error_blr_cd = {df: [] for df in data_fractions}
        start_bow = time.time()
        print('Training for BOW Model')
        #Run the Logistic Regression for 30 iteration for each training set size
        for epochs in range(1, 31):
                print("For Epoch")
                print(f"{epochs:-^20}")
                for data_fraction in data_fractions:
                    a_blr_error_bow, _, _, _ = bow_algo.BayesianLogisticRegression(data_fraction)
                    a_epochs_error_blr_bow[data_fraction].append(a_blr_error_bow)
        end_bow = (time.time() - start_bow)

        print('Training for LDA- Model')
        start_cd = time.time()
        for epochs in range(1, 31):
                print("For Epoch")
                print(f"{epochs:-^20}")
                for data_fraction in data_fractions:
                    a_blr_error_cd, _, _, _ = Cd_algo.BayesianLogisticRegression(data_fraction)
                    a_epochs_error_blr_cd[data_fraction].append(a_blr_error_cd)
        end_cd = time.time() - start_cd
        return a_epochs_error_blr_bow,a_epochs_error_blr_cd,end_bow,end_cd
    
    def plot_error(self,bow_error: dict, cd_error: dict, title: str):
        mean_bow_dict = {key: sum(vals) / len(vals) for key, vals in bow_error.items()}
        std_bow_dict = {key: np.std(np.array(vals)) for key, vals in bow_error.items()}
        mean_cd_dict = {key: sum(vals) / len(vals) for key, vals in cd_error.items()}
        std_cd_dict = {key: np.std(np.array(vals)) for key, vals in cd_error.items()}

        fig = plt.figure()
        ax = fig.add_subplot(111)
        X = mean_bow_dict.keys()
        ax.set_title(f"For dataset {title}")
        ax.errorbar(
            X, mean_bow_dict.values(), std_bow_dict.values(), label="Bag-of-Words"
        )
        ax.errorbar(
            X, mean_cd_dict.values(), std_cd_dict.values(), label="LDA-Topic-Representation"
        )
        ax.set_ylabel("Error rate")
        ax.set_xlabel("Fraction of Data Set Size")
        ax.legend(loc="best")
        fig.savefig(f"{title}.png")
        plt.show()
