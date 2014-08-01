from __future__ import division
import numpy as np
from GaussianHMM import GaussianHMM
from evaluation import *
import os

DATA_DIR = "./Dataset/"
ACC_DIR = "./Dataset/"
REG_PARAM = 3 # Controls the strength of regularization. (Lambda parameter)
rgzn_modes = GaussianHMM.RgznModes()

def learn_and_predict_MAP(data, K):
    model = GaussianHMM(K, data, rgzn_modes.MAP_SCALE_FREE)
    final_param = REG_PARAM
    model.learn(data, zeta=final_param, init=True)    
    return np.array(model.decode(data), dtype="int8")

def learn_and_predict_INERTIAL(data, K):    
    model = GaussianHMM(K, data, rgzn_modes.INERTIAL)
    final_param = REG_PARAM
    model.learn(data, zeta=final_param, init=True)    
    return np.array(model.decode(data), dtype="int8")

def do_quantitative(mode = rgzn_modes.MAP_SCALE_FREE, results_dir = "./Results/"): 
    evaluation_dir = results_dir + "/evaluation/"
    if not os.path.exists(evaluation_dir):
        os.makedirs(evaluation_dir)
    results = []
    learn_func = learn_and_predict_INERTIAL if mode == rgzn_modes.INERTIAL \
                 else learn_and_predict_MAP

    suffix = "INERTIAL" if mode == rgzn_modes.INERTIAL \
             else "MAP"
    for i in range(100):
        try:
            data = np.loadtxt(ACC_DIR + "./human_acc_data_10000_%d.dat" % i)
            true_states = np.loadtxt(ACC_DIR + "./human_acc_states_10000_%d.dat" % i, dtype="int8")
            K = len(set(true_states))        
            predicted_states = learn_func(data, K)
            pt = np.hstack((np.array(predicted_states).reshape((-1,1)), np.array(true_states).reshape((-1,1))))
            np.savetxt(evaluation_dir + "./%s_%d_indv_predictions.txt" % (suffix, i), pt, fmt="%d")
            ents = entropies(predicted_states, true_states)
            normed_voi = (ents[0] + ents[1] - 2 * ents[3]) / ents[2]
            res = (max_correct(predicted_states, true_states), segment_measurements(predicted_states, true_states), normed_voi)
            results.append(res)
            print i, res
        except:        
            print "************* Example %d Failed" % i
            break
    accuracies = [r[0][1] for r in results]
    actual_ratios = [r[1][0] for r in results]
    divergence_ratios = [r[1][1] for r in results]
    diff_in_num_of_segments = [r[1][2] for r in results]
    perfect_segmentations = [r[0][2] for r in results]
    vois = [r[2] for r in results]
    print "Avg. Accuracy", np.mean(accuracies)
    print "Avg. Segment Number Ratio", np.mean(actual_ratios)
    print "Avg. Segment Number Divergence Ratio", np.mean(divergence_ratios)
    print "Avg. Number of Segments Difference", np.mean(diff_in_num_of_segments)
    print "Total Number of Perfect Segmentations", np.sum(perfect_segmentations), "of", i + 1
    print "Avg. Normed Variation of Information", np.mean(vois)
    out = open(results_dir + "./results_%s.txt" % suffix, "w")
    output = ("Avg. Accuracy %.3f \n" + \
              "Avg. Segment Number Ratio %.2f \n" + \
              "Avg. Segment Number Divergence Ratio %.2f\n" + \
              "Avg. Number of Segments Difference %.2f\n" + \
              "Total Number of Perfect Segmentations %d \n" + \
              "Avg. Variation of Information %.2f") % (np.mean(accuracies),
                                                     np.mean(actual_ratios),
                                                     np.mean(divergence_ratios),
                                                     np.mean(diff_in_num_of_segments), 
                                                     np.sum(perfect_segmentations),
                                                     np.mean(vois))
    out.write(output)
    all_results = np.vstack((accuracies,
                             actual_ratios,
                             divergence_ratios,
                             diff_in_num_of_segments,
                             perfect_segmentations,
                             vois)).T
    np.savetxt(results_dir + "./quant_analysis_%s.txt" % suffix, all_results)


def main():
    """ Performs quantitative analysis on 100 time series dataset, for both regularization modes """
    do_quantitative(mode=rgzn_modes.INERTIAL, results_dir='./Results_INERTIAL_%d/' % REG_PARAM)
    do_quantitative(mode=rgzn_modes.MAP_SCALE_FREE, results_dir='./Results_MAP_%d/' % REG_PARAM)
    
if __name__ == "__main__":
    main()

