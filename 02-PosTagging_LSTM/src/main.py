from POSTaggerTrainer import POSTaggerTrainer
from POSTaggerTester import POSTaggerTester
from LSTMPOSTagger import LSTMPOSTagger
import sys
import os

# Dynamic import
sys.path.append(os.getcwd())
from homework2 import ModelIO

# Directory
dataDir = os.path.join(os.getcwd(), 'data')
outputDir = os.path.join(os.getcwd(), 'output')

# Data
trainPath = os.path.join(dataDir, 'en-ud-train.conllu')
testPath = os.path.join(dataDir, 'en-ud-test.conllu')
modelPath = os.path.join(outputDir, 'model-reg2.keras')

# Train
trainer = POSTaggerTrainer()
model = trainer.train(trainPath)


# Tagger
lstm_tagger = LSTMPOSTagger(model)
lstm_tagger.load_resources()
# Test
tester = POSTaggerTester()
score = tester.test(lstm_tagger, testPath)
print(score)


# Save model
#ModelIO.save(model, modelPath)
