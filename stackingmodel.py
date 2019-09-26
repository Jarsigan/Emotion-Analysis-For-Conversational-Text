import numpy as np
import json, argparse, os
import re
import io
from keras.layers.core import Activation
from keras import initializers as initializations
from keras import regularizers, constraints
from keras.layers.merge import add, average, concatenate, maximum, multiply
from sklearn.svm import SVC
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Reshape, Dense, Dropout, Embedding, CuDNNGRU, Bidirectional, GRU, Input, Flatten, SpatialDropout1D, LSTM, Convolution1D
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Embedding
from keras import optimizers
from keras.callbacks import EarlyStopping
import fasttext
from keras.models import load_model



# Path to training and testing data file. This data can be downloaded from a link, details of which will be provided.
trainDataPath = ""
testDataPath = ""
# Output file that will be generated. This file can be directly submitted.
solutionPath = ""
# Path to directory where GloVe file is saved.
gloveDir = ""
# Path to directory where fasttext file is saved.
fasttextDir = ""

NUM_FOLDS = None                   # Value of K in K-fold Cross Validation
NUM_CLASSES = None                 # Number of classes - Happy, Sad, Angry, Others
MAX_NB_WORDS = None                # To set the upper limit on the number of tokens extracted using keras.preprocessing.text.Tokenizer
MAX_SEQUENCE_LENGTH = None         # All sentences having lesser number of words than this will be padded
EMBEDDING_DIM = None               # The dimension of the word embeddings
BATCH_SIZE = None                  # The batch size to be chosen for training the model.
LSTM_DIM = None                    # The dimension of the representations learnt by the LSTM model
DROPOUT = None                     # Fraction of the units to drop for the linear transformation of the inputs. Ref - https://keras.io/layers/recurrent/
NUM_EPOCHS = None                  # Number of epochs to train a model for

hidden_dim = 120
kernel_size = 3
nb_filter = 60


label2emotion = {0:"others", 1:"happy", 2: "sad", 3:"angry"}
emotion2label = {"others":0, "happy":1, "sad":2, "angry":3}


def preprocessData(dataFilePath, mode):
    """Load data from a file, process and return indices, conversations and labels in separate lists
    Input:
        dataFilePath : Path to train/test file to be processed
        mode : "train" mode returns labels. "test" mode doesn't return labels.
    Output:
        indices : Unique conversation ID list
        conversations : List of 3 turn conversations, processed and each turn separated by the <eos> tag
        labels : [Only available in "train" mode] List of labels
    """
    indices = []
    conversations = []
    labels = []
    u1 = []
    u2 = []
    u3 = []
    with io.open(dataFilePath, encoding="utf8") as finput:
        finput.readline()
        for line in finput:
            # Convert multiple instances of . ? ! , to single instance
            # okay...sure -> okay . sure
            # okay???sure -> okay ? sure
            # Add whitespace around such punctuation
            # okay!sure -> okay ! sure
            repeatedChars = ['.', '?', '!', ',']
            for c in repeatedChars:
                lineSplit = line.split(c)
                while True:
                    try:
                        lineSplit.remove('')
                    except:
                        break
                cSpace = ' ' + c + ' '
                line = cSpace.join(lineSplit)

            line = line.strip().split('\t')
            if mode == "train":
                # Train data contains id, 3 turns and label
                label = emotion2label[line[4]]
                labels.append(label)

            conv = ' <eos> '.join(line[1:4])

            u1.append(line[1])
            u2.append(line[2])
            u3.append(line[3])

            # Remove any duplicate spaces
            duplicateSpacePattern = re.compile(r'\ +')
            conv = re.sub(duplicateSpacePattern, ' ', conv)

            indices.append(int(line[0]))
            conversations.append(conv.lower())

    if mode == "train":
        # return indices, conversations, labels, u1, u2, u3
        return indices, conversations, labels
    else:
        #
        return indices, conversations


def getMetrics(predictions, ground):
    """Given predicted labels and the respective ground truth labels, display some metrics
    Input: shape [# of samples, NUM_CLASSES]
        predictions : Model output. Every row has 4 decimal values, with the highest belonging to the predicted class
        ground : Ground truth labels, converted to one-hot encodings. A sample belonging to Happy class will be [0, 1, 0, 0]
    Output:
        accuracy : Average accuracy
        microPrecision : Precision calculated on a micro level. Ref - https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin/16001
        microRecall : Recall calculated on a micro level
        microF1 : Harmonic mean of microPrecision and microRecall. Higher value implies better classification
    """
    # [0.1, 0.3 , 0.2, 0.1] -> [0, 1, 0, 0]
    discretePredictions = to_categorical(predictions.argmax(axis=1))

    truePositives = np.sum(discretePredictions*ground, axis=0)
    falsePositives = np.sum(np.clip(discretePredictions - ground, 0, 1), axis=0)
    falseNegatives = np.sum(np.clip(ground-discretePredictions, 0, 1), axis=0)

    print("True Positives per class : ", truePositives)
    print("False Positives per class : ", falsePositives)
    print("False Negatives per class : ", falseNegatives)

    # ------------- Macro level calculation ---------------
    macroPrecision = 0
    macroRecall = 0
    # We ignore the "Others" class during the calculation of Precision, Recall and F1
    for c in range(1, NUM_CLASSES):
        precision = truePositives[c] / (truePositives[c] + falsePositives[c])
        macroPrecision += precision
        recall = truePositives[c] / (truePositives[c] + falseNegatives[c])
        macroRecall += recall
        f1 = ( 2 * recall * precision ) / (precision + recall) if (precision+recall) > 0 else 0
        print("Class %s : Precision : %.3f, Recall : %.3f, F1 : %.3f" % (label2emotion[c], precision, recall, f1))

    macroPrecision /= 3
    macroRecall /= 3
    macroF1 = (2 * macroRecall * macroPrecision ) / (macroPrecision + macroRecall) if (macroPrecision+macroRecall) > 0 else 0
    print("Ignoring the Others class, Macro Precision : %.4f, Macro Recall : %.4f, Macro F1 : %.4f" % (macroPrecision, macroRecall, macroF1))

    # ------------- Micro level calculation ---------------
    truePositives = truePositives[1:].sum()
    falsePositives = falsePositives[1:].sum()
    falseNegatives = falseNegatives[1:].sum()

    print("Ignoring the Others class, Micro TP : %d, FP : %d, FN : %d" % (truePositives, falsePositives, falseNegatives))

    microPrecision = truePositives / (truePositives + falsePositives)
    microRecall = truePositives / (truePositives + falseNegatives)

    microF1 = ( 2 * microRecall * microPrecision ) / (microPrecision + microRecall) if (microPrecision+microRecall) > 0 else 0
    # -----------------------------------------------------

    predictions = predictions.argmax(axis=1)
    ground = ground.argmax(axis=1)
    accuracy = np.mean(predictions==ground)

    print("Accuracy : %.4f, Micro Precision : %.4f, Micro Recall : %.4f, Micro F1 : %.4f" % (accuracy, microPrecision, microRecall, microF1))
    return accuracy, microPrecision, microRecall, microF1


def writeNormalisedData(dataFilePath, texts):
    """Write normalised data to a file
    Input:
        dataFilePath : Path to original train/test file that has been processed
        texts : List containing the normalised 3 turn conversations, separated by the <eos> tag.
    """
    normalisedDataFilePath = dataFilePath.replace(".txt", "_normalised.txt")
    with io.open(normalisedDataFilePath, 'w', encoding='utf8') as fout:
        with io.open(dataFilePath, encoding='utf8') as fin:
            fin.readline()
            for lineNum, line in enumerate(fin):
                line = line.strip().split('\t')
                normalisedLine = texts[lineNum].strip().split('<eos>')
                fout.write(line[0] + '\t')
                # Write the original turn, followed by the normalised version of the same turn
                fout.write(line[1] + '\t' + normalisedLine[0] + '\t')
                fout.write(line[2] + '\t' + normalisedLine[1] + '\t')
                fout.write(line[3] + '\t' + normalisedLine[2] + '\t')
                try:
                    # If label information available (train time)
                    fout.write(line[4] + '\n')
                except:
                    # If label information not available (test time)
                    fout.write('\n')

def getMetrics(predictions, ground):
    """Given predicted labels and the respective ground truth labels, display some metrics
    Input: shape [# of samples, NUM_CLASSES]
        predictions : Model output. Every row has 4 decimal values, with the highest belonging to the predicted class
        ground : Ground truth labels, converted to one-hot encodings. A sample belonging to Happy class will be [0, 1, 0, 0]
    Output:
        accuracy : Average accuracy
        microPrecision : Precision calculated on a micro level. Ref - https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin/16001
        microRecall : Recall calculated on a micro level
        microF1 : Harmonic mean of microPrecision and microRecall. Higher value implies better classification
    """
    # [0.1, 0.3 , 0.2, 0.1] -> [0, 1, 0, 0]
    discretePredictions = to_categorical(predictions.argmax(axis=1))

    truePositives = np.sum(discretePredictions * ground, axis=0)
    falsePositives = np.sum(np.clip(discretePredictions - ground, 0, 1), axis=0)
    falseNegatives = np.sum(np.clip(ground - discretePredictions, 0, 1), axis=0)

    print("True Positives per class : ", truePositives)
    print("False Positives per class : ", falsePositives)
    print("False Negatives per class : ", falseNegatives)

    # ------------- Macro level calculation ---------------
    macroPrecision = 0
    macroRecall = 0
    # We ignore the "Others" class during the calculation of Precision, Recall and F1
    for c in range(1, NUM_CLASSES):
        precision = truePositives[c] / (truePositives[c] + falsePositives[c])
        macroPrecision += precision
        recall = truePositives[c] / (truePositives[c] + falseNegatives[c])
        macroRecall += recall
        f1 = (2 * recall * precision) / (precision + recall) if (precision + recall) > 0 else 0
        print("Class %s : Precision : %.3f, Recall : %.3f, F1 : %.3f" % (label2emotion[c], precision, recall, f1))

    macroPrecision /= 3
    macroRecall /= 3
    macroF1 = (2 * macroRecall * macroPrecision) / (macroPrecision + macroRecall) if (
                                                                                             macroPrecision + macroRecall) > 0 else 0
    print("Ignoring the Others class, Macro Precision : %.4f, Macro Recall : %.4f, Macro F1 : %.4f" % (
        macroPrecision, macroRecall, macroF1))

    # ------------- Micro level calculation ---------------
    truePositives = truePositives[1:].sum()
    falsePositives = falsePositives[1:].sum()
    falseNegatives = falseNegatives[1:].sum()

    print(
        "Ignoring the Others class, Micro TP : %d, FP : %d, FN : %d" % (truePositives, falsePositives, falseNegatives))

    microPrecision = truePositives / (truePositives + falsePositives)
    microRecall = truePositives / (truePositives + falseNegatives)

    microF1 = (2 * microRecall * microPrecision) / (microPrecision + microRecall) if (
                                                                                             microPrecision + microRecall) > 0 else 0
    # -----------------------------------------------------

    predictions = predictions.argmax(axis=1)
    ground = ground.argmax(axis=1)
    accuracy = np.mean(predictions == ground)

    print("Accuracy : %.4f, Micro Precision : %.4f, Micro Recall : %.4f, Micro F1 : %.4f" % (
        accuracy, microPrecision, microRecall, microF1))
    return accuracy, microPrecision, microRecall, microF1



def getGloveEmbeddingMatrix(wordIndex, dim):
    """Populate an embedding matrix using a word-index. If the word "happy" has an index 19,
       the 19th row in the embedding matrix should contain the embedding vector for the word "happy".
    Input:
        wordIndex : A dictionary of (word : index) pairs, extracted using a tokeniser
    Output:
        embeddingMatrix : A matrix where every row has 100 dimensional GloVe embedding
    """
    embeddingsIndex = {}
    # Load the embedding vectors from ther GloVe file
    # with io.open(os.path.join(gloveDir, 'glove.twitter.27B.200d.txt'), encoding="utf8") as f:
    with io.open(os.path.join(gloveDir, 'datastories.twitter.300d.txt'), encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            embeddingVector = np.asarray(values[1:], )
            embeddingsIndex[word] = embeddingVector

    print('Found %s word vectors.' % len(embeddingsIndex))

    # Minimum word index of any word is 1.
    embeddingMatrix = np.zeros((len(wordIndex) + 1, dim))
    for word, i in wordIndex.items():
        embeddingVector = embeddingsIndex.get(word)
        if embeddingVector is not None:
            # words not found in embedding index will be all-zeros.
            embeddingMatrix[i] = embeddingVector

    return embeddingMatrix


def getEmbeddingByBin(wordIndex, dim):
    model = fasttext.load_model("250D_322M_tweets.bin")
    embeddingMatrix = np.zeros((len(wordIndex) + 1, dim))
    for word, i in wordIndex.items():
        v = model.get_word_vector(word)
        embeddingMatrix[i] = v
    return embeddingMatrix


def build_CNNLSTM_Model(embeddingMatrix, embedding_dim, hidden_dim, name):
    """Constructs the architecture of the model
    Input:
        embeddingMatrix : The embedding matrix to be loaded in the embedding layer.
    Output:
        model : A basic CNN-LSTM model
    """
    # Convolution parameters
    filter_length = 3
    nb_filter = 150
    pool_length = 2
    cnn_activation = 'relu'
    border_mode = 'same'

    # RNN parameters
    output_size = 50
    rnn_activation = 'tanh'
    recurrent_activation = 'hard_sigmoid'

    # Compile parameters
    loss = 'binary_crossentropy'
    optimizer = 'rmsprop'

    embeddingLayer = Embedding(embeddingMatrix.shape[0],
                               embedding_dim,
                               weights=[embeddingMatrix],
                               input_length=MAX_SEQUENCE_LENGTH,
                               trainable=False)
    model = Sequential()
    model.add(embeddingLayer)
    model.add(Dropout(0.5))
    model.add(Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode=border_mode,
                            activation=cnn_activation,
                            subsample_length=1))
    model.add(MaxPooling1D(pool_length=pool_length))
    model.add(LSTM(output_dim=output_size, activation=rnn_activation, recurrent_activation=recurrent_activation))
    model.add(Dropout(0.25))
    model.add(Dense(NUM_CLASSES))
    model.add(Activation('sigmoid'))
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model, name


def build_CNNLSTM_Model_Concat(embeddingMatrix, embedding_dim, hidden_dim, name):
    # Convolution parameters
    filter_length = 3
    nb_filter = 256
    pool_length = 2
    cnn_activation = 'relu'
    border_mode = 'same'

    # RNN parameters
    output_size = 50
    rnn_activation = 'tanh'
    recurrent_activation = 'hard_sigmoid'

    # Compile parameters
    loss = 'binary_crossentropy'
    optimizer = 'rmsprop'

    x1 = Input(shape=(MAX_SEQUENCE_LENGTH,), sparse=False, dtype='int32', name='main_input1')

    e0 = Embedding(embeddingMatrix.shape[0],
                   embedding_dim,
                   weights=[embeddingMatrix],
                   input_length=MAX_SEQUENCE_LENGTH,
                   trainable=False)
    emb1 = e0(x1)
    d=Dropout(0.2)
    o=d(emb1)
    c0 = Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode=border_mode,
                            activation=cnn_activation,
                            subsample_length=1)(o)
    # p0 = {'max': GlobalMaxPooling1D()(c0), 'avg': GlobalAveragePooling1D()(c0)}.get(pooling, c0)
    print(c0.get_shape())

    p0=MaxPooling1D(pool_length=pool_length)(c0)
    print(p0.get_shape())
    # p0=Reshape((1))(p0
    # emb1=Flatten()(emb1)
    # p0=Flatten()(p0)
    # print(p0.get_shape())
    # p0 = concatenate([emb1, p0])
    # p0=Reshape((1,))(p0)
    lstm = Bidirectional(LSTM(dropout=0.2,output_dim=output_size))
    out = lstm(p0)
    opt = Dense(NUM_CLASSES, activation='sigmoid')(out)
    dropout=Dropout(0.25)(opt)
    model = Model([x1], dropout)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model, name

def get_stacking(clf, data, labels, x_dev, x_test, n_folds=1, name=None):
    
    train_num, test_num, dev_num = data.shape[0], x_test.shape[0], x_dev.shape[0]
    second_level_train_set = np.zeros((train_num, 4))
    test_result = np.zeros((test_num, 4))
    dev_result = np.zeros((dev_num, 4))
    test_nfolds_sets = []
    dev_nfolds_stes = []

    for k in range(NUM_FOLDS):
        print('-'*80)
        print('Fold %d/%d  %s' %(k+1, NUM_FOLDS, name))
        validationSize = int(len(data)/NUM_FOLDS)
        index1 = validationSize * k
        index2 = validationSize * (k + 1)

        xTrain = np.vstack((data[:index1], data[index2:]))
        yTrain = np.vstack((labels[:index1], labels[index2:]))
        xVal = data[index1:index2]
        yVal = labels[index1:index2]
        print("Building model...")
        early_stopping = EarlyStopping(monitor='val_acc', patience=10)
        clf.fit(xTrain, yTrain, validation_data=[xVal, yVal], epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, verbose=2,
                callbacks=[early_stopping])
        path = './model/{0}'.format(name)
        if not os.path.exists(path):
            os.makedirs(path)
        clf.save('./model/%s/bi-%s-model-fold-%d.h5' % (name, name, k))

        second_level_train_set[index1:index2] = clf.predict(xVal)
        dev_nfolds_stes.append(clf.predict(x_dev))
        test_nfolds_sets.append(clf.predict(x_test))

    for item in test_nfolds_sets:
        test_result += item
    test_result = test_result / n_folds

    for item in dev_nfolds_stes:
        dev_result += item
    dev_result = dev_result / n_folds

    # second_level_test_set[:] = test_nfolds_sets.mean(axis=1)
    return second_level_train_set, dev_result, test_result

def calculatef1(model, endPath, testData, name):
    metrics = {"accuracy": [],
               "microPrecision": [],
               "microRecall": [],
               "microF1": []}

    solutionPath = './model/%s/%s-test.txt' % (name, name)
    predictions = model.predict(testData, batch_size=BATCH_SIZE)
    predictions = predictions.argmax(axis=1)
    with io.open(solutionPath, "w", encoding="utf8") as fout:
        fout.write('\t'.join(["id", "turn1", "turn2", "turn3", "label"]) + '\n')
        with io.open(testDataPath, encoding="utf8") as fin:
            fin.readline()
            for lineNum, line in enumerate(fin):
                fout.write('\t'.join(line.strip().split('\t')[:4]) + '\t')
                fout.write(label2emotion[predictions[lineNum]] + '\n')
    print("Completed. Model parameters: ")
    print("Learning rate : %.3f, LSTM Dim : %d, Dropout : %.3f, Batch_size : %d"
          % (LEARNING_RATE, LSTM_DIM, DROPOUT, BATCH_SIZE))

    print("Calculating F1 value")
    # solIndices, solTexts, sollabels = preprocessData(endPath, mode="train")
    solIndices, solTexts, sollabels = preprocessData(endPath, mode="train")
    sollabels = to_categorical(np.asarray(sollabels))

    # endIndices, endTexts, endlabels = preprocessData(solutionPath, mode="train")
    endIndices, endTexts, endlabels = preprocessData(solutionPath, mode="train")
    endlabels = to_categorical(np.asarray(endlabels))

    # predictions = model.predict(xVal, batch_size=BATCH_SIZE)
    accuracy, microPrecision, microRecall, microF1 = getMetrics(endlabels, sollabels)
    metrics["accuracy"].append(accuracy)
    metrics["microPrecision"].append(microPrecision)
    metrics["microRecall"].append(microRecall)
    metrics["microF1"].append(microF1)
    print(metrics)
    return metrics

def get_stacking_model(embeddinglist, traindata, trainlabels, testData, endPath):
    members = []
    submodelMetrics = []
    # build_CNNLSTM_Model(embeddinglist[1], embedding_dim=300, hidden_dim=150, name='Glovecnnlstm')
    # for clf, name in [build_CNNLSTM_Model(embeddinglist[0], embedding_dim=250, hidden_dim=400, name='fasttextcnnlstm'),
    #                   build_CNNLSTM_Model_Concat(embeddinglist[0], embedding_dim=250, hidden_dim=400, name='fasttextcnnlstmoptimized')]:
        # print("Building model...")
        # clf.fit(traindata, trainlabels, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
        # path = './model/{0}'.format(name)
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # clf.save('./model/%s/bi-%s-model-fold-%d.h5' % (name, name, 1))
        # members.append(clf)
        # #sub model f1 prediction
        # submodelpredictionmetrics = calculatef1(clf, endPath, testData, name)
        # submodelMetrics.append(submodelpredictionmetrics)
        # model = load_model('./model/%s/bi-%s-model-fold-%d.h5' % (name, name, 1))

        # members.append(model)
    members.append(load_model('./model/fasttextcnnlstm/EP20_LR300e-5_LDim200_BS200.h5'))
    members.append(load_model('./model/fasttextcnnlstm/EP20_LR300e-5_LDim200_BS200.h5'))
    # update all layers in all models to not be trainable
    for i in range(len(members)):
        model = members[i]
        for layer in model.layers:
            # make not trainable
            layer.trainable = False
            # rename to avoid 'unique layer name' issue
            layer.name = 'ensemble_' + str(i + 1) + '_' + layer.name

    # define multi-headed input
    ensemble_visible = [model.input for model in members]
    # concatenate merge output from each model
    ensemble_outputs = [model.output for model in members]
    merge = concatenate(ensemble_outputs)
    hidden = Dense(10, activation='relu')(merge)
    # inp = Reshape((1, 10,))(hidden)
    # print(inp.shape)
    output = Dense(4, activation='softmax')(hidden)
    stackingmodel = Model(inputs=ensemble_visible, outputs=output)
    stackingmodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(stackingmodel.summary())
    return stackingmodel


def main():
    parser = argparse.ArgumentParser(description="Baseline Script for SemEval")
    parser.add_argument('-config', help='Config to read details', required=True, default='testBaseline.config')
    args = parser.parse_args()

    with open(args.config) as configfile:
        config = json.load(configfile)

    global trainDataPath, devDataPath, testDataPath, solutionPath, gloveDir
    global NUM_FOLDS, NUM_CLASSES, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM
    global BATCH_SIZE, LSTM_DIM, DROPOUT, NUM_EPOCHS, LEARNING_RATE

    trainDataPath = config["train_data_path"]
    devDataPath = config["dev_data_path"]
    testDataPath = config["test_data_path"]
    endPath = config["standard_data_path"]

    solutionPath = config["solution_path"]
    gloveDir = config["glove_dir"]

    NUM_FOLDS = config["num_folds"]
    NUM_CLASSES = config["num_classes"]
    MAX_NB_WORDS = config["max_nb_words"]
    MAX_SEQUENCE_LENGTH = config["max_sequence_length"]
    # EMBEDDING_DIM = config["embedding_dim"]
    BATCH_SIZE = config["batch_size"]
    LSTM_DIM = config["lstm_dim"]
    DROPOUT = config["dropout"]
    LEARNING_RATE = config["learning_rate"]
    NUM_EPOCHS = config["num_epochs"]

    print("Processing training data...")
    trainIndices, trainTexts, labels = preprocessData(trainDataPath, mode="train")
    # Write normalised text to file to check if normalisation works. Disabled now. Uncomment following line to enable
    # writeNormalisedData(trainDataPath, trainTexts)
    print("Processing dev data...")
    devIndices, devTexts, devlabels = preprocessData(devDataPath, mode="train")

    print("Processing test data...")
    testIndices, testTexts = preprocessData(testDataPath, mode="test")
    # writeNormalisedData(testDataPath, testTexts)

    print("Extracting tokens...")
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(trainTexts)
    # devtokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    # devtokenizer.fit_on_texts(devTexts)
    # testtokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    # testtokenizer.fit_on_texts(testTexts)

    # converts text to vector form of word subcript
    trainSequences = tokenizer.texts_to_sequences(trainTexts)
    devSequences = tokenizer.texts_to_sequences(devTexts)
    testSequences = tokenizer.texts_to_sequences(testTexts)
    data = pad_sequences(trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
    devdata = pad_sequences(devSequences, maxlen=MAX_SEQUENCE_LENGTH)
    testData = pad_sequences(testSequences, maxlen=MAX_SEQUENCE_LENGTH)

    labels = to_categorical(np.asarray(labels))
    devlabels = to_categorical(np.asarray(devlabels))
    # pickle.dump(testData, open('testData.pickle', 'wb'))

    # save the number for each word, starting at 1
    # #{'some': 1, 'thing': 2,'to': 3 ','eat': 4, drink': 5}
    wordIndex = tokenizer.word_index
    print("Found %s unique tokens." % len(wordIndex))

    print("Populating embedding matrix...")
    gloveEmbeddingMatrix = getGloveEmbeddingMatrix(wordIndex, dim=300)
    # elmoEmbeddingMatrix = getElmoEmbeddingMatrix(wordIndex, dim=1024)
    fasttextEmbeddingMatrix = getEmbeddingByBin(wordIndex, dim=250)


    print("Shape of training data tensor: ", data.shape)
    print("Shape of label tensor: ", labels.shape)

    # Randomize data
    # arr = np.arange(len(y_smote_resampled))
    # data = np.array(x_smote_resampled)[arr]
    # labels = np.array(y_smote_resampled)[arr]
    # print(len(data))

    # suffle traindata
    np.random.shuffle(trainIndices)
    data = data[trainIndices]
    labels = labels[trainIndices]

    # suffle dev data
    np.random.shuffle(devIndices)
    devdata = devdata[devIndices]
    devlabels = devlabels[devIndices]

    metrics = {"accuracy": [],
               "microPrecision": [],
               "microRecall": [],
               "microF1": []}

    print('K-fold start')
    print('-'*60)

    """ ,
                      lstmModel(fasttextEmbeddingMatrix, embedding_dim=250, hidden_dim=400, name='fasttextLstm'),
                      gruModel(fasttextEmbeddingMatrix, embedding_dim=250, hidden_dim=400, name='fasttextGru'),
                      lstmModel(gloveEmbeddingMatrix, embedding_dim=300, hidden_dim=150, name='GloveLstm'),
                      gruModel(gloveEmbeddingMatrix, embedding_dim=300, hidden_dim=150, name='GloveGru'),
                      ,
                      build_CNNLSTM_Model(gloveEmbeddingMatrix, embedding_dim=300, hidden_dim=150, name='Glovecnnlstm')
                      """

    embeddinglist = []
    embeddinglist.append(fasttextEmbeddingMatrix)
    embeddinglist.append(gloveEmbeddingMatrix)
    # for clf, name in [build_CNNLSTM_Model(fasttextEmbeddingMatrix, embedding_dim=250, hidden_dim=400, name='fasttextcnnlstm')]:
    #     train_set, dev_set, test_set = get_stacking(clf, data, labels, devData, testData, name=name)
    #     train_sets.append(train_set)
    #     dev_sets.append(dev_set)
    #     test_sets.append(test_set)
    #
    # meta_train = np.concatenate([result_set.reshape(-1, 4) for result_set in train_sets], axis=1)
    # meta_dev = np.concatenate([dev_result_set.reshape(-1, 4) for dev_result_set in dev_sets], axis=1)
    # meta_test = np.concatenate([y_test_set.reshape(-1, 4) for y_test_set in test_sets], axis=1)
    # path = './stacking_new_elmo.pickle'
    # pickle.dump([meta_train, meta_dev, meta_test, labels], open(path, 'wb'))
    #
    # svc = SVC(kernel='sigmoid', gamma=1.3, C=4)
    # svc.fit(meta_train, np.array(labels.argmax(axis=1)))
    # predictions = svc.predict(meta_test)
    # predictions = predictions.argmax(axis=1)

    stackingmodel = get_stacking_model(embeddinglist, data, labels, testData, endPath)

    # prepare input data
    X = [devdata for _ in range(len(stackingmodel.input))]
    # encode output data
    inputy_enc = to_categorical(devlabels)
    # fit model
    stackingmodel.fit(X, devlabels, epochs=5, batch_size=BATCH_SIZE)
    stackingmodel.save('stackingmodel%d.h5' % (300))
    # model = load_model('EP%d_LR%de-5_LDim%d_BS%d.h5'%(NUM_EPOCHS, int(LEARNING_RATE*(10**5)), LSTM_DIM, BATCH_SIZE))

    print("Creating solution file...")

    testData = pad_sequences(testSequences, maxlen=MAX_SEQUENCE_LENGTH)

    testinput = [testData for _ in range(len(stackingmodel.input))]
    predictions = stackingmodel.predict(testinput, batch_size=BATCH_SIZE)
    predictions = predictions.argmax(axis=1)

    with io.open(solutionPath, "w", encoding="utf8") as fout:
        fout.write('\t'.join(["id", "turn1", "turn2", "turn3", "label"]) + '\n')
        with io.open(testDataPath, encoding="utf8") as fin:
            fin.readline()
            for lineNum, line in enumerate(fin):
                fout.write('\t'.join(line.strip().split('\t')[:4]) + '\t')
                fout.write(label2emotion[predictions[lineNum]] + '\n')
    print("Completed. Model parameters: ")
    print("Learning rate : %.3f, LSTM Dim : %d, Dropout : %.3f, Batch_size : %d"
          % (LEARNING_RATE, LSTM_DIM, DROPOUT, BATCH_SIZE))

    print("Calculating F1 value")
    # solIndices, solTexts, sollabels = preprocessData(endPath, mode="train")
    solIndices, solTexts, sollabels = preprocessData(endPath, mode="train")
    sollabels = to_categorical(np.asarray(sollabels))

    # endIndices, endTexts, endlabels = preprocessData(solutionPath, mode="train")
    endIndices, endTexts, endlabels = preprocessData(solutionPath, mode="train")
    endlabels = to_categorical(np.asarray(endlabels))

    # predictions = model.predict(xVal, batch_size=BATCH_SIZE)
    accuracy, microPrecision, microRecall, microF1 = getMetrics(endlabels, sollabels)
    metrics["accuracy"].append(accuracy)
    metrics["microPrecision"].append(microPrecision)
    metrics["microRecall"].append(microRecall)
    metrics["microF1"].append(microF1)

    print(metrics)


if __name__ == '__main__':
    main()
