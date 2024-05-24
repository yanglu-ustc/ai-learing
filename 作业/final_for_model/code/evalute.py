import numpy as np

SAMPLE_RATE = 8000


def evaluate(data_length, label_data, predict_data):
    voice_length = 0
    predict_voice_length = 0

    label = np.full(data_length, 0)
    for i in range(len(label_data)):
        a = int(label_data[i][0])
        b = int(label_data[i][1])
        label[a:b + 1] = 1
        voice_length += (b - a)

    predict = np.full(data_length, 0)
    for i in range(len(predict_data)):
        a = int(predict_data[i][0])
        b = int(predict_data[i][1])
        predict[a:b + 1] = 1
        predict_voice_length += (b - a)

    false_detection = 0
    miss_detection = 0
    acc = 0
    tp = 0
    for i in range(data_length):
        if label[i] == 0 and predict[i] == 1:
            false_detection += 1
        if label[i] == 1 and predict[i] == 0:
            miss_detection += 1
        if label[i] == predict[i]:
            acc += 1
        if label[i] == 1 and predict[i] == 1:
            tp += 1

    accuracy = acc / data_length
    recall = tp / voice_length
    precision = tp / predict_voice_length
    f1_score = (2 * precision * recall) / (precision + recall)

    return f1_score, accuracy, recall, precision
