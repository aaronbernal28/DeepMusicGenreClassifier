# DeepMusicGenreClassifier

Este informe aborda la clasificación de géneros musicales utilizando el dataset GTZAN (10 géneros, 100 canciones de 30 segundos cada uno). El objetivo es analizar y clasificar la música empleando diversas arquitecturas de redes neuronales. Se Implementa métodos que incluyen el procesamiento secuencial RNNs, el análisis de espectrogramas (Mel, MFCC, CQT) con CNNs (1D), y la utilización de redes neuronales pre entrenadas para la extracción de features relevantes (transfer learning) como Wav2vec2 y EnCodecMAE. Como criterios de evaluación se emplean la entropía cruzada y la precisión (accuracy) para determinar cuál de estas arquitecturas ofrece un mejor desempeño en la tarea de clasificación.

Dataset: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data
