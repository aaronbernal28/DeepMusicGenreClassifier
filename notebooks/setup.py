import torch
from sklearn.model_selection import train_test_split
import numpy as np
import librosa
import soundfile as sf
import time
import torchaudio
from torch.utils.data import Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader

# Configuración del dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")
torch.cuda.manual_seed(28)
torch.cuda.set_per_process_memory_fraction(0.8)

# Funciones auxiliares
def train_test_split_path():
    '''
    Division fija de los datos de entrenamiento y testeo para todos los experimentos
    '''
    X_path = []
    y = []
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    for id, genre in enumerate(genres):
        for i in range(100):
            if not (i == 54 and genre == 'jazz'):
                path = f'../Data/genres_original/{genre}/{genre}.000{i:02}.wav'
                X_path.append(path)
                y.append(id)
    X_train_path, X_eval_path, y_train, y_eval = train_test_split(X_path, y, test_size=0.1, random_state=28, stratify=y)
    X_train_path, X_test_path, y_train, y_test = train_test_split(X_train_path, y_train, test_size=0.2, random_state=28, stratify=y_train)
    return X_train_path, X_test_path, X_eval_path, y_train, y_test, y_eval

_, _, _, y_train, y_test, y_eval = train_test_split_path()

def train(model, train_dataloader, val_dataloader, optimizer, criterion, NUM_EPOCHS):
    train_losses = []
    val_losses = []

    print("Iniciando entrenamiento...")
    print("-" * 50)
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        # Entrenamiento
        model.train()
        train_loss = 0
        n = 0

        for batch_idx, batch in enumerate(train_dataloader):
            input = batch['input']
            target = batch['target']

            optimizer.zero_grad()

            output = model.forward(input)
            loss = criterion(output, target)

            loss.backward()

            # Gradient clipping para evitar exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()
            n += 1

        train_loss /= n
        train_losses.append(train_loss)

        # Validación
        model.eval()
        val_loss = 0
        m = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                input = batch['input']
                target = batch['target']

                output = model.forward(input)
                loss = criterion(output, target)
                val_loss += loss.item()
                m += 1
        val_loss /= m
        val_losses.append(val_loss)

        if (epoch + 1) % 5 == 0:
            print(f'Época {epoch+1}/{NUM_EPOCHS}')
            print(f'  Pérdida Entrenamiento: {train_loss:.4f}')
            print(f'  Pérdida Testeo: {val_loss:.4f}')
            print(f'  {"Mejorando" if val_loss < min(val_losses[:-1] + [float("inf")]) else "Empeorando"}')
    
    end_time = time.time()
    print(f"Entrenamiento completado! Tiempo total: {end_time - start_time:.2f} segundos")
    return train_losses, val_losses

def accuracy(model, dataloader):
    model.eval()
    accur = 0
    with torch.no_grad():
        for item in dataloader:
            input = item['input'].unsqueeze(0)
            target = item['target']

            output = model.forward(input)
            pred = torch.nn.functional.one_hot(torch.argmax(output), 10)
            accur += int(sum(pred == target) == 10)

    accur /= len(dataloader)
    return accur

def mel_spectrogram(x, sr, max_len, hop_length, win_length):
    melspec = librosa.feature.melspectrogram(y=x[:max_len], sr=sr, hop_length=hop_length, win_length=win_length, n_mels=80)
    melspec_db = librosa.power_to_db(melspec, ref=np.max)
    return melspec_db

def mfcc_combined(x, sr, max_len, hop_length, win_length):
    mfcc = librosa.feature.mfcc(y = x[:max_len], sr=sr, n_mfcc=20, n_fft=win_length, n_mels=128, dct_type=2, norm='ortho', center=False)
    mfcc_delta = librosa.feature.delta(mfcc, order=1)
    mfcc_delta_delta = librosa.feature.delta(mfcc, order=2)
    mfcc_combined = np.concatenate((mfcc, mfcc_delta, mfcc_delta_delta), axis=0)
    mfcc_combined_norm = (mfcc_combined - np.mean(mfcc_combined, axis=1, keepdims=True)) / np.std(mfcc_combined, axis=1, keepdims=True)
    return mfcc_combined_norm

def CQT(x, sr, max_len, hop_length, win_length):
    C = np.abs(librosa.cqt(x[:max_len], sr=sr))
    C_db = librosa.amplitude_to_db(C, ref=np.max)
    return C_db

def Wav2Vec2_features(x, sr, max_len, model, new_sr=16000):
    waveform = torch.tensor(x[:max_len], dtype=torch.float32).to(device)
    waveform = torchaudio.functional.resample(waveform, sr, new_sr)
    
    with torch.inference_mode():
        features, _ = model.extract_features(waveform.unsqueeze(0))

    return features[-1].squeeze().detach().cpu().numpy()

def get_features(paths, feature_name, sr, max_len, hop_length, win_length):
    X = []

    if feature_name == 'wav2vec2':
        bundle = torchaudio.pipelines.HUBERT_BASE
        modelWav = bundle.get_model().to(device)
        modelWav.eval()
        for path in paths:
            x, _ = sf.read(path)
            X.append(Wav2Vec2_features(x, sr, max_len, modelWav))

    else:

        if feature_name == 'mel':
            feature = mel_spectrogram
        elif feature_name == 'mfcc':
            feature = mfcc_combined
        elif feature_name == 'cqt':
            feature = CQT
        else: 
            raise ValueError("Feature no soportado. Usa 'wav2vec2', 'mel', 'mfcc' o 'cqt'")
        
        for path in paths:
            x, _ = sf.read(path)
            X.append(feature(x, sr=sr, max_len=max_len, hop_length=hop_length, win_length=win_length))

    return X

def plot_losses(train_losses, val_losses, ax, log_scale=False, title=None, accuracy=None):
    ax.plot(train_losses, label='Train', c='darkcyan')
    ax.plot(val_losses, label='Test', c='darkred')
    ax.set_xlabel('Epocas')
    ax.set_ylabel('Loss')
    if accuracy is not None:
        full_title = f"{title} (Accur: {accuracy:.2%})" if title else f"Accur: {accuracy:.2%}"
    else:
        full_title = title
    ax.set_title(full_title)
    if log_scale:
        ax.set_yscale('log')
    ax.legend()
    ax.grid(True)

class Sequence_dataset(Dataset):
    def __init__(self, X, y):
        self.pairs = list(zip(X, y))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        xs, ys = self.pairs[idx]

        return {
            'input': torch.tensor(xs.T).to(device=device, dtype=torch.float32),
            'target': F.one_hot(torch.tensor(ys), 10).to(device=device, dtype=torch.float32),
            'input_length': xs.T.shape,
            'target_length': 10
        }
    
class Image_dataset(Dataset):
    def __init__(self, X, y):
        self.pairs = list(zip(X, y))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        xs, ys = self.pairs[idx]

        return {
            'input': torch.tensor(xs).to(device=device, dtype=torch.float32),
            'target': F.one_hot(torch.tensor(ys), 10).to(device=device, dtype=torch.float32),
            'input_length': xs.shape,
            'target_length': 10
        }
    
def get_data_max(feature_name, dataset_class):
    X_train = np.load(f'../Data/X_train_{feature_name}.npy') 
    X_test = np.load(f'../Data/X_test_{feature_name}.npy') 

    print(f"Entrenamiento: {len(X_train)} pares")
    print(f"Testeo: {len(X_test)} pares")
    print("-" * 50)

    if feature_name == 'EnCodecMAE':
        # venia (719, 1, 1031, 768) -> (719, 1031, 768) -> (719, 768, 1031) ~ (N, seq_len, timesteps)
        X_train = X_train.squeeze(1).transpose(0, 2, 1)
        X_test = X_test.squeeze(1).transpose(0, 2, 1)

    if feature_name == 'wav2vec2':
        # venia (719, 1, 1031, 768) -> (719, 1031, 768) -> (719, 768, 1031) ~ (N, seq_len, timesteps)
        X_train = X_train.transpose(0, 2, 1)
        X_test = X_test.transpose(0, 2, 1)

    train_dataset = dataset_class(X_train, y_train)
    test_dataset = dataset_class(X_test, y_test)

    print('Primer elemento del dataset de entrenamiento:', train_dataset[0])
    print("-" * 50)

    return train_dataset, test_dataset

def train_pro_max(model_class, train_dataset, test_dataset, feature_name: str, num_epoths: dict, batch_sizes: dict, learning_rates: dict, optimizer_name = 'Adam', **params):
    '''
    model_class es la clase del modelo a entrenar
    feature_name es mel o mfcc o cqt o wav2vec2 o EnCodecMAE
    '''
    BATCH_SIZE = batch_sizes[feature_name]
    LEARNING_RATE = learning_rates[feature_name]
    NUM_EPOCHS = num_epoths[feature_name]

    # Inicializa el modelo 
    model = model_class(**params).to(device)

    # Función de pérdida y optimizador
    criterion = nn.CrossEntropyLoss()
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    elif optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    else:
        raise ValueError(f"Optimizador {optimizer_name} no soportado")
    
    print(f"Modelo creado con {sum(p.numel() for p in model.parameters())} parámetros")
    print("-" * 50)
    #### Entrenamiento del modelo #####

    # DataLoaders en batches
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    train_losses, val_losses = train(model, train_dataloader, val_dataloader, optimizer, criterion, NUM_EPOCHS)

    return model, train_losses, val_losses