from NLP import Morph
import sklearn_crfsuite as crf
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import pickle

MAX_DELTA = 12

# Extra point - Train extra + Train.eng
EXTRA = dict(NAME='Extra model', TRAIN='task_extra/train_extra.txt', TEST='task1_data/test.eng.txt',
             DEV='task1_data/dev.eng.txt', NAME_PLOT='Extra model to changes delta', SAVE_MODEL='extra_model.model')

# English
ENG = dict(NAME='English model', TRAIN='task1_data/train.eng.txt', TEST='task1_data/test.eng.txt',
           DEV='task1_data/dev.eng.txt', NAME_PLOT='English model to changes delta', SAVE_MODEL='crf_model.model')

# English
ENG2 = dict(NAME='English model 2', TRAIN='task1_data/train_dev.eng.txt', TEST='task1_data/test.eng.txt',
            DEV='task1_data/dev.eng.txt', NAME_PLOT='English model to changes delta 2', SAVE_MODEL='crf_model2.model')

# Italian
ITA = dict(NAME='Italian model 2', TRAIN='task2_data/train.it.txt', TEST='task2_data/test.it.txt',
           DEV='task2_data/dev.it.txt', NAME_PLOT='Italian model to changes delta', SAVE_MODEL='ita_model.model')

# Italian
ITA2 = dict(NAME='Italian model 2', TRAIN='task2_data/train_dev.it.txt', TEST='task2_data/test.it.txt',
            DEV='task2_data/dev.it.txt', NAME_PLOT='Italian model to changes delta 2', SAVE_MODEL='ita_model2.model')

''' Optimize delta '''


# show_tabulate for see the tag score

def optimize_delta(train, test, show_tabulate=True):
    score = []
    models = []
    f1_max = 0.
    f1_max_index = 0
    for i in range(1, MAX_DELTA + 1):
        print('Delta:' + str(i))
        model = Morph(i, train, test)
        c = crf.CRF(algorithm='ap', max_iterations=100)
        c.fit(model.training_feature, model.training_label)
        prediction = c.predict(model.testing_feature)
        metrics = total_score(model.testing_label, prediction)
        score.append(metrics)
        models.append(c)
        if score[i - 1]['AVG'][2] > f1_max:
            f1_max = score[i - 1]['AVG'][2]
            f1_max_index = i - 1

    return score, [f1_max_index, f1_max], models


''' Score for single TAG '''


def single_tag_score(true_tag_set, pred_tag_set, TAG):
    TP = 0  # True Positive
    FP = 0  # False Positive
    FN = 0  # False Negative
    SUP = 0  # Support
    for true_condition, predicted_condition in zip(true_tag_set, pred_tag_set):
        for trueTag, predTag in zip(true_condition, predicted_condition):

            if TAG is trueTag:
                SUP += 1
                if TAG is predTag:
                    TP += 1
                else:
                    FN += 1

            if TAG is predTag:
                if TAG is not trueTag:
                    FP += 1

    P = TP / (TP + FP)
    R = TP / (TP + FN)
    F1 = (2 * P * R) / (P + R)

    return {TAG: [P, R, F1, SUP]}


''' Score for set of TAGS '''


def total_score(true_tag_set, pred_tag_set):
    tag_set = ['B', 'M', 'E', 'S']
    avg_P = 0
    avg_R = 0
    avg_F = 0
    total_SUPP = 0
    scores = dict()
    for i in range(len(tag_set)):
        scores.update(single_tag_score(true_tag_set, pred_tag_set, tag_set[i]))

    for key, metric in scores.items():
        avg_P += metric[0] * metric[3]
        avg_R += metric[1] * metric[3]
        avg_F += metric[2] * metric[3]
        total_SUPP += metric[3]

    avg_P = avg_P / total_SUPP
    avg_R = avg_R / total_SUPP
    avg_F = avg_F / total_SUPP
    scores.update({'AVG': [avg_P, avg_R, avg_F, total_SUPP]})

    return scores


''' PLOT the result'''


def plot_result(scores, name_plot):
    ind = np.arange(1, len(scores) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.grid(axis='y')
    plt.title('Scores')
    plot_precision_recall_f1 = []
    for i in range(len(scores)):
        plot_precision_recall_f1.append(scores[i]['AVG'])
    plot_precision_recall_f1 = np.asarray(plot_precision_recall_f1)
    ax.plot(ind, plot_precision_recall_f1[:, 0], c='green', label='F1', linewidth=0.5)
    ax.plot(ind, plot_precision_recall_f1[:, 1],
            c='red', label='Precision', linewidth=0.5)
    ax.plot(ind, plot_precision_recall_f1[:, 2], c='blue', label='Recall', linewidth=0.5)
    ax.legend(loc='lower right')
    plt.savefig(name_plot + '.png', format='png', dpi=200)


''' Save model with pickle '''


def save_model(name_model, model):
    pickle.dump(model, open(name_model, "wb"))


''' Run '''


def run(dict_model, show_tabulate=True):
    print('Running model: ', dict_model['NAME'])
    scores, best_f1, models = optimize_delta(
        dict_model['TRAIN'], dict_model['TEST'], show_tabulate=True)
    best_f1_value = best_f1[1]
    best_f1_delta = best_f1[0]
    print('Best F1:{} for delta:{}'.format(best_f1_value, best_f1_delta + 1))
    if show_tabulate:
        print(tabulate(scores[best_f1_delta], headers="keys", tablefmt="grid") + '\n')
    plot_result(scores, 'images/' + dict_model['NAME_PLOT'])
    save_model('models/' + dict_model['SAVE_MODEL'], models[best_f1_delta])


# Run for the Extra model
run(EXTRA)

# Run for the English model
# run(ENG)

# Train + Dev
# run(ENG2)

# Run for the Italian model
# run(ITA)

# Run for the Italian model
# run(ITA2)
