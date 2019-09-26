# Please use python 3.5 or above
import numpy as np
from keras.engine import Layer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras_self_attention import SeqSelfAttention
from keras.utils import to_categorical

import fasttext

from keras.layers import Dense, Input, Flatten, Reshape, concatenate, Dropout, Convolution1D, Activation, GRU, \
    SimpleRNN, GlobalMaxPooling1D, GlobalAveragePooling1D, Concatenate
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Embedding
from keras.layers import LSTM, Bidirectional
from keras.models import Model
from keras import optimizers
from keras import regularizers

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras import optimizers
from keras.models import load_model
import json, argparse, os
import re
import io
import sys

# Path to training and testing data file. This data can be downloaded from a link, details of which will be provided.
from keras_preprocessing import sequence
from tensorflow.python.estimator import keras

trainDataPath = ""
testDataPath = ""
# Output file that will be generated. This file can be directly submitted.
solutionPath = ""
# Path to directory where fasttext file is saved.
fasttextDir = ""
# Path to directory where deepmoji file is saved.
deepmojiDir = ""

NUM_FOLDS = None  # Value of K in K-fold Cross Validation
NUM_CLASSES = None  # Number of classes - Happy, Sad, Angry, Others
MAX_NB_WORDS = None  # To set the upper limit on the number of tokens extracted using keras.preprocessing.text.Tokenizer
MAX_SEQUENCE_LENGTH = None  # All sentences having lesser number of words than this will be padded
WORD_EMBEDDING_DIM = None  # The dimension of the word embeddings
CONTEXTUAL_EMBEDDING_DIM = None  # The dimension of the contextual embeddings
BATCH_SIZE = None  # The batch size to be chosen for training the model.
LSTM_DIM = None  # The dimension of the representations learnt by the LSTM model
DROPOUT = None  # Fraction of the units to drop for the linear transformation of the inputs. Ref - https://keras.io/layers/recurrent/
NUM_EPOCHS = None  # Number of epochs to train a model for

label2emotion = {0: "others", 1: "happy", 2: "sad", 3: "angry"}
emotion2label = {"others": 0, "happy": 1, "sad": 2, "angry": 3}


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
        return indices, conversations, labels, u1, u2, u3
    else:
        return indices, conversations, u1, u2, u3


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


def getEmbeddingMatrix(wordIndex):
    """Populate an embedding matrix using a word-index. If the word "happy" has an index 19,
       the 19th row in the embedding matrix should contain the embedding vector for the word "happy".
    Input:
        wordIndex : A dictionary of (word : index) pairs, extracted using a tokeniser
    Output:
        embeddingMatrix : A matrix where every row has 100 dimensional GloVe embedding
    """
    embeddingsIndex = {}
    # Load the embedding vectors from ther GloVe file
    with io.open(os.path.join(fasttextDir, 'lexsub_words.txt'), encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            # print(values[1:])
            embeddingVector = np.asarray(values[1:], dtype='float32')
            embeddingsIndex[word] = embeddingVector

    print('Found %s word vectors.' % len(embeddingsIndex))

    # Minimum word index of any word is 1.
    embeddingMatrix = np.zeros((len(wordIndex) + 1, CONTEXTUAL_EMBEDDING_DIM))
    for word, i in wordIndex.items():
        embeddingVector = embeddingsIndex.get(word)
        if embeddingVector is not None:
            # words not found in embedding index will be all-zeros.
            embeddingMatrix[i] = embeddingVector

    return embeddingMatrix


def getEmbeddingByBin(wordIndex):
    model = fasttext.load_model("250D_322M_tweets.bin")
    embeddingMatrix = np.zeros((len(wordIndex) + 1, WORD_EMBEDDING_DIM))
    for word, i in wordIndex.items():
        v = model.get_word_vector(word)
        embeddingMatrix[i] = v
    return embeddingMatrix

def buildBilstmAttention(wordEmbeddingMatrix):

    input1 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='main_input1')
    input2 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='main_input2')
    input3 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='main_input3')
    embd = Embedding(wordEmbeddingMatrix.shape[0],
                               WORD_EMBEDDING_DIM,
                               weights=[wordEmbeddingMatrix],
                               input_length=MAX_SEQUENCE_LENGTH,
                               trainable=False)
    emb1 = embd(input1)
    emb2 = embd(input2)
    emb3 = embd(input3)
    bilstm = Bidirectional(LSTM(LSTM_DIM, return_sequences=True))
    lstm1 = bilstm(emb1)
    lstm2 = bilstm(emb2)
    lstm3 = bilstm(emb3)
    inp = Concatenate(axis=-1)([lstm1, lstm2, lstm3])
    att = SeqSelfAttention(attention_activation='sigmoid')(inp)
    flattened = Flatten()(att)
    print(att.shape)
    out = Dense(NUM_CLASSES, activation='softmax')(flattened)
    print(out.shape)
    model = Model(inputs=[input1, input2, input3], outputs=out)
    adam = optimizers.adam(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['acc'])

    print(model.summary())

    return model

def build_BiLSTMCNNwithSelfAttention_Model(wordEmbeddingMatrix):
    filters = 250
    pooling = 'max'

    input1 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='main_input1')
    input2 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='main_input2')
    input3 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='main_input3')
    embd = Embedding(wordEmbeddingMatrix.shape[0],
                     WORD_EMBEDDING_DIM,
                     weights=[wordEmbeddingMatrix],
                     input_length=MAX_SEQUENCE_LENGTH,
                     trainable=False)
    emb1 = embd(input1)
    emb2 = embd(input2)
    emb3 = embd(input3)

    lstm = LSTM(LSTM_DIM, return_sequences=True)
    lstm1 = lstm(emb1)
    lstm2 = lstm(emb2)
    lstm3 = lstm(emb3)
    # print(lstm.shape)
    inp = Concatenate(axis=-1)([lstm1, lstm2, lstm3])
    att = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       kernel_regularizer=regularizers.l2(1e-4),
                       bias_regularizer=regularizers.l1(1e-4),
                       attention_regularizer_weight=1e-4,
                       name='Attention')(inp)

    c0 = Conv1D(filters, 1, padding='valid', activation='relu', strides=1)(att)
    p0 = {'max': GlobalMaxPooling1D()(c0), 'avg': GlobalAveragePooling1D()(c0)}.get(pooling, c0)
    print(p0.shape)
    out = Reshape((1, filters,))(p0)
    flattened = Flatten()(out)
    opt = Dense(NUM_CLASSES, activation='softmax')(flattened)

    model = Model([input1,input2,input3], opt)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

def build_LSTMCNN_Model_New(embeddingMatrix):
    # Convolution parameters
    filter_length = 3
    filters = 150
    pool_length = 2
    cnn_activation = 'relu'
    border_mode = 'same'
    pooling = 'max'

    # RNN parameters
    output_size = 50
    rnn_activation = 'tanh'
    recurrent_activation = 'hard_sigmoid'

    # Compile parameters
    loss = 'binary_crossentropy'
    optimizer = 'rmsprop'

    input1 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='main_input1')
    input2 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='main_input2')
    input3 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='main_input3')

    embd = Embedding(embeddingMatrix.shape[0],
                               WORD_EMBEDDING_DIM,
                               weights=[embeddingMatrix],
                               input_length=MAX_SEQUENCE_LENGTH,
                               trainable=False)

    emb1 = embd(input1)
    emb2 = embd(input2)
    emb3 = embd(input3)

    lstm = LSTM(LSTM_DIM, return_sequences=True)
    lstm1 = lstm(emb1)
    lstm2 = lstm(emb2)
    lstm3 = lstm(emb3)
    # print(lstm.shape)
    inp = Concatenate(axis=-1)([lstm1, lstm2, lstm3])

    c0 = Conv1D(filters, 1, padding='valid', activation='relu', strides=1)(inp)
    p0 = {'max': GlobalMaxPooling1D()(c0), 'avg': GlobalAveragePooling1D()(c0)}.get(pooling, c0)
    print(p0.shape)
    # out = Reshape((1, filters,))(p0)
    opt = Dense(NUM_CLASSES, activation='softmax')(p0)

    model = Model([input1, input2, input3], opt)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    return model

def build_CNNLSTM_Model_New(embeddingMatrix):
    # Convolution parameters
    filter_length = 3
    nb_filter = 150
    pool_length = 2
    cnn_activation = 'relu'
    border_mode = 'same'
    pooling = 'max'

    # RNN parameters
    output_size = 50
    rnn_activation = 'tanh'
    recurrent_activation = 'hard_sigmoid'

    # Compile parameters
    loss = 'binary_crossentropy'
    optimizer = 'rmsprop'

    input1 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='main_input1')
    input2 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='main_input2')
    input3 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='main_input3')

    embd = Embedding(embeddingMatrix.shape[0],
                               WORD_EMBEDDING_DIM,
                               weights=[embeddingMatrix],
                               input_length=MAX_SEQUENCE_LENGTH,
                               trainable=False)

    emb1 = embd(input1)
    emb2 = embd(input2)
    emb3 = embd(input3)

    # inp = Concatenate(axis=-1)([emb1, emb2, emb3])

    convolution = Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode=border_mode,
                            activation=cnn_activation,
                            subsample_length=1)

    pooling = MaxPooling1D(pool_length=pool_length)

    # con1 = convolution(emb1)
    # con2 = convolution(emb2)
    # con3 = convolution(emb3)

    pol1 = pooling(emb1)
    pol2 = pooling(emb2)
    pol3 = pooling(emb3)

    inp = Concatenate(axis=-1)([pol1, pol2, pol3])
    lstm = LSTM(LSTM_DIM, return_sequences=False)
    lstm1 = lstm(inp)
    # lstm2 = lstm(pol2)
    # lstm3 = lstm(pol3)
    print(lstm1.shape)

    # inp = Concatenate(axis=-1)([lstm1, lstm2, lstm3])
    # print(inp.shape)
    # out = Reshape((1, 3*LSTM_DIM,))(inp)
    # lstm_up = LSTM(LSTM_DIM, return_sequences=False)
    # lstm_upf = lstm_up(out)

    # out = Reshape((1, LSTM_DIM,))(inp)

    opt = Dense(NUM_CLASSES, activation='softmax')(lstm1)

    model = Model([input1, input2, input3], opt)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    return model

def build_CNNLSTM_Model(embeddingMatrix):
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
                               WORD_EMBEDDING_DIM,
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

    return model


def main():
    parser = argparse.ArgumentParser(description="Baseline Script for SemEval")
    parser.add_argument('-config', help='Config to read details', required=True)
    args = parser.parse_args()

    with open(args.config) as configfile:
        config = json.load(configfile)

    global trainDataPath, testDataPath, solutionPath, fasttextDir, deepmojiDir
    global NUM_FOLDS, NUM_CLASSES, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, WORD_EMBEDDING_DIM, CONTEXTUAL_EMBEDDING_DIM
    global BATCH_SIZE, LSTM_DIM, DROPOUT, NUM_EPOCHS, LEARNING_RATE

    trainDataPath = config["train_data_path"]
    testDataPath = config["test_data_path"]
    solutionPath = config["solution_path"]
    endPath = config["standard_data_path"]
    fasttextDir = config["fasttext_dir"]
    deepmojiDir = config["deepmoji_dir"]

    NUM_FOLDS = config["num_folds"]
    NUM_CLASSES = config["num_classes"]
    MAX_NB_WORDS = config["max_nb_words"]
    MAX_SEQUENCE_LENGTH = config["max_sequence_length"]
    WORD_EMBEDDING_DIM = config["word_embedding_dim"]
    CONTEXTUAL_EMBEDDING_DIM = config["contextual_embedding_dim"]
    BATCH_SIZE = config["batch_size"]
    LSTM_DIM = config["lstm_dim"]
    DROPOUT = config["dropout"]
    LEARNING_RATE = config["learning_rate"]
    NUM_EPOCHS = config["num_epochs"]

    print("Processing training data...")
    # trainIndices, trainTexts, labels = preprocessData(trainDataPath, mode="train")
    trainIndices, trainTexts, labels, u1_train, u2_train, u3_train = preprocessData(trainDataPath, mode="train")
    # Write normalised text to file to check if normalisation works. Disabled now. Uncomment following line to enable
    # writeNormalisedData(trainDataPath, trainTexts)
    print("Processing test data...")
    # testIndices, testTexts = preprocessData(testDataPath, mode="test")
    testIndices, testTexts, u1_test, u2_test, u3_test = preprocessData(testDataPath, mode="test")
    # writeNormalisedData(testDataPath, testTexts)

    print("Extracting tokens...")
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(u1_train+u2_train+u3_train)
    # trainSequences = tokenizer.texts_to_sequences(trainTexts)
    # testSequences = tokenizer.texts_to_sequences(testTexts)
    u1_trainSequences, u2_trainSequences, u3_trainSequences = tokenizer.texts_to_sequences(
        u1_train), tokenizer.texts_to_sequences(u2_train), tokenizer.texts_to_sequences(u3_train)
    u1_testSequences, u2_testSequences, u3_testSequences = tokenizer.texts_to_sequences(
        u1_test), tokenizer.texts_to_sequences(u2_test), tokenizer.texts_to_sequences(u3_test)

    wordIndex = tokenizer.word_index
    print("Found %s unique tokens." % len(wordIndex))

    print("Populating embedding matrix...")

    wordEmbeddingMatrix = getEmbeddingByBin(wordIndex)
    u1_data = pad_sequences(u1_trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
    u2_data = pad_sequences(u2_trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
    u3_data = pad_sequences(u3_trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = to_categorical(np.asarray(labels))
    # Randomize data
    np.random.shuffle(trainIndices)
    u1_data = u1_data[trainIndices]
    u2_data = u2_data[trainIndices]
    u3_data = u3_data[trainIndices]


    labels = labels[trainIndices]

    # Perform k-fold cross validation
    metrics = {"accuracy": [],
               "microPrecision": [],
               "microRecall": [],
               "microF1": []}
    print("\n======================================")

    print("Retraining model on entire data to create solution file")
    # model = buildBilstmAttention(wordEmbeddingMatrix)
    model = build_CNNLSTM_Model_New(wordEmbeddingMatrix)
    # model = build_BiLSTMCNNwithSelfAttention_Model(wordEmbeddingMatrix)
    model.fit([u1_data,u2_data,u3_data], labels, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

    model.save('EP%d_LR%de-5_LDim%d_BS%d.h5' % (NUM_EPOCHS, int(LEARNING_RATE * (10 ** 5)), LSTM_DIM, BATCH_SIZE))
    # model = load_model('EP%d_LR%de-5_LDim%d_BS%d.h5'%(NUM_EPOCHS, int(LEARNING_RATE*(10**5)), LSTM_DIM, BATCH_SIZE))

    print("Creating solution file...")
    u1_testData, u2_testData, u3_testData = pad_sequences(u1_testSequences, maxlen=MAX_SEQUENCE_LENGTH), pad_sequences(
        u2_testSequences, maxlen=MAX_SEQUENCE_LENGTH), pad_sequences(u3_testSequences, maxlen=MAX_SEQUENCE_LENGTH)
    predictions = model.predict([u1_testData, u2_testData, u3_testData], batch_size=BATCH_SIZE)
    predictions = predictions.argmax(axis=1)


    # predictions = predictions.argmax(axis=1)

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
    solIndices, solTexts, sollabels, u1_sol, u2_sol, u3_sol = preprocessData(endPath, mode="train")
    sollabels = to_categorical(np.asarray(sollabels))

    # endIndices, endTexts, endlabels = preprocessData(solutionPath, mode="train")
    endIndices, endTexts, endlabels, u1_end, u2_end, u3_end = preprocessData(solutionPath, mode="train")
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
