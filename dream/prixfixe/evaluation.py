import numpy as np
import pandas as pd
import json
from collections import OrderedDict
import csv
import os
from scipy.stats import pearsonr, spearmanr

def load_ground_truth(filename):
    with open(filename) as f:
        reader = csv.reader(f, delimiter="\t")
        lines = list(reader)
    expressions = [float(line[1]) for line in lines]
    return np.array(expressions)

def load_promoter_class_indices(file_path):
    df = pd.read_csv(file_path)
    if 'pos' in df.columns:
        return np.unique(np.array(df['pos']))
    elif 'alt_pos' in df.columns and 'ref_pos' in df.columns:
        SNVs_alt = list(df['alt_pos'])
        SNVs_ref = list(df['ref_pos'])
        return list(set(list(zip(SNVs_alt, SNVs_ref))))

def load_public_leaderboard_single_indices(file_path):
    with open(file_path, 'r') as f:
        indices = [int(indice) for indice in list(json.load(f).keys())]
    return indices

def load_public_leaderboard_double_indices(file_path):
    with open(file_path, 'r') as f:
        return [(int(indice.split(',')[0]), int(indice.split(',')[1])) for indice in list(json.load(f).keys())]

def calculate_correlations(index_list, expressions, GROUND_TRUTH_EXP):
    PRED_DATA = OrderedDict()
    GROUND_TRUTH = OrderedDict()

    for j in index_list:
        PRED_DATA[str(j)] = float(expressions[j])
        GROUND_TRUTH[str(j)] = float(GROUND_TRUTH_EXP[j])

    pearson = pearsonr(list(GROUND_TRUTH.values()), list(PRED_DATA.values()))[0]
    spearman = spearmanr(list(GROUND_TRUTH.values()), list(PRED_DATA.values()))[0]

    return pearson, spearman


def calculate_diff_correlations(pair_list, expressions, GROUND_TRUTH_EXP):
    Y_pred_selected = []
    expressions_selected = []

    for pair in pair_list:
        ref, alt = pair[0], pair[1]
        Y_pred_selected.append(expressions[alt] - expressions[ref])
        expressions_selected.append(GROUND_TRUTH_EXP[alt] - GROUND_TRUTH_EXP[ref])

    Y_pred_selected = np.array(Y_pred_selected)
    expressions_selected = np.array(expressions_selected)

    pearson = pearsonr(expressions_selected, Y_pred_selected)[0]
    spearman = spearmanr(expressions_selected, Y_pred_selected)[0]

    return pearson, spearman

def evaluate_predictions(expressions, discard_public_leaderboard_indices=True):
    
    expressions = np.array(expressions)
    GROUND_TRUTH_EXP = load_ground_truth('data/filtered_test_data_with_MAUDE_expression.txt')
    # Load indices for different promoter classes
    high = load_promoter_class_indices('data/test_subset_ids/high_exp_seqs.csv')
    low = load_promoter_class_indices('data/test_subset_ids/low_exp_seqs.csv')
    yeast = load_promoter_class_indices('data/test_subset_ids/yeast_seqs.csv')
    random = load_promoter_class_indices('data/test_subset_ids/all_random_seqs.csv')
    challenging = load_promoter_class_indices('data/test_subset_ids/challenging_seqs.csv')
    SNVs = load_promoter_class_indices('data/test_subset_ids/all_SNVs_seqs.csv')
    motif_perturbation = load_promoter_class_indices('data/test_subset_ids/motif_perturbation_seqs.csv')
    motif_tiling = load_promoter_class_indices('data/test_subset_ids/motif_tiling_seqs.csv')

    if discard_public_leaderboard_indices:
        # Load indices used in the public leaderboard
        public_high = load_public_leaderboard_single_indices('data/public_leaderboard_ids/high_exp_indices.json')
        public_low = load_public_leaderboard_single_indices('data/public_leaderboard_ids/low_exp_indices.json')
        public_yeast = load_public_leaderboard_single_indices('data/public_leaderboard_ids/yeast_exp_indices.json')
        public_random = load_public_leaderboard_single_indices('data/public_leaderboard_ids/random_exp_indices.json')
        public_challenging = load_public_leaderboard_single_indices('data/public_leaderboard_ids/challenging_exp_indices.json')
        public_SNVs = load_public_leaderboard_double_indices('data/public_leaderboard_ids/SNVs_exp_indices.json')
        public_motif_perturbation = load_public_leaderboard_double_indices('data/public_leaderboard_ids/motif_perturbation_exp_indices.json')
        public_motif_tiling = load_public_leaderboard_double_indices('data/public_leaderboard_ids/motif_tiling_exp_indices.json')
        public_single_indices = public_high + public_low + public_yeast + public_random + public_challenging
        public_double_indices = public_SNVs + public_motif_perturbation + public_motif_tiling

        public_indices = []

        for indice in public_double_indices:
            public_indices.append(indice[0])
            public_indices.append(indice[1])

        for indice in public_single_indices:
            public_indices.append(indice)

        public_indices = list(set(public_indices))
        high = [exp for exp in high if exp not in public_indices]
        low = [exp for exp in low if exp not in public_indices]
        yeast = [exp for exp in yeast if exp not in public_indices]
        random = [exp for exp in random if exp not in public_indices]
        challenging = [exp for exp in challenging if exp not in public_indices]
        SNVs = [exp for exp in SNVs if exp not in public_double_indices]
        motif_perturbation = [exp for exp in motif_perturbation if exp not in public_double_indices]
        motif_tiling = [exp for exp in motif_tiling if exp not in public_double_indices]
        final_all = [exp for exp in list(range(len(GROUND_TRUTH_EXP))) if exp not in public_indices]
    else:
        final_all = list(range(len(GROUND_TRUTH_EXP)))

    # Calculate correlations
    pearson, spearman = calculate_correlations(final_all, expressions, GROUND_TRUTH_EXP)
    high_pearson, high_spearman = calculate_correlations(high, expressions, GROUND_TRUTH_EXP)
    low_pearson, low_spearman = calculate_correlations(low, expressions, GROUND_TRUTH_EXP)
    yeast_pearson, yeast_spearman = calculate_correlations(yeast, expressions, GROUND_TRUTH_EXP)
    random_pearson, random_spearman = calculate_correlations(random, expressions, GROUND_TRUTH_EXP)
    challenging_pearson, challenging_spearman = calculate_correlations(challenging, expressions, GROUND_TRUTH_EXP)

    # Calculate difference correlations
    SNVs_pearson, SNVs_spearman = calculate_diff_correlations(SNVs, expressions, GROUND_TRUTH_EXP)
    motif_perturbation_pearson, motif_perturbation_spearman = calculate_diff_correlations(motif_perturbation, expressions, GROUND_TRUTH_EXP)
    motif_tiling_pearson, motif_tiling_spearman = calculate_diff_correlations(motif_tiling, expressions, GROUND_TRUTH_EXP)


    # Calculate scores
    pearsons_score = (pearson**2 + 0.3 * high_pearson**2 + 0.3 * low_pearson**2 + 0.3 * yeast_pearson**2 + 
                    0.3 * random_pearson**2 + 0.5 * challenging_pearson**2 + 1.25 * SNVs_pearson**2 + 
                    0.3 * motif_perturbation_pearson**2 + 0.4 * motif_tiling_pearson**2) / 4.65


    spearmans_score = (spearman + 0.3 * high_spearman + 0.3 * low_spearman + 0.3 * yeast_spearman 
                    + 0.3 * random_spearman + 0.5 * challenging_spearman + 1.25 * SNVs_spearman
                    + 0.3 * motif_perturbation_spearman + 0.4 * motif_tiling_spearman) / 4.65

    # Print scores
    print('******************************************************')
    print('Pearson Score: {}\n'.format(pearsons_score))
    print('Spearman Score: {}\n'.format(spearmans_score))
    print('******************************************************')
    print('all r: {}\n'.format(pearson))
    print('all r\u00b2: {}\n'.format(pearson**2))
    print('all \u03C1: {}\n'.format(spearman))
    print('******************************************************')
    print('high r: {}\n'.format(high_pearson))
    print('low r: {}\n'.format(low_pearson))
    print('yeast r: {}\n'.format(yeast_pearson))
    print('random r: {}\n'.format(random_pearson))
    print('challenging r: {}\n'.format(challenging_pearson))
    print('SNVs r: {}\n'.format(SNVs_pearson))
    print('motif perturbation r: {}\n'.format(motif_perturbation_pearson))
    print('motif tiling r: {}\n'.format(motif_tiling_pearson))
    print('******************************************************')
    print('high \u03C1: {}\n'.format(high_spearman))
    print('low \u03C1: {}\n'.format(low_spearman))
    print('yeast \u03C1: {}\n'.format(yeast_spearman))
    print('random \u03C1: {}\n'.format(random_spearman))
    print('challenging \u03C1: {}\n'.format(challenging_spearman))
    print('SNVs \u03C1: {}\n'.format(SNVs_spearman))
    print('motif perturbation \u03C1: {}\n'.format(motif_perturbation_spearman))
    print('motif tiling \u03C1: {}\n'.format(motif_tiling_spearman))
    print('******************************************************')