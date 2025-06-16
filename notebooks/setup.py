import torch
from sklearn.model_selection import train_test_split
import numpy as np
import librosa
import soundfile as sf
import time

# Configuración del dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")
torch.cuda.manual_seed(28)
torch.cuda.set_per_process_memory_fraction(0.6)

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
            print(f'  Pérdida Validación: {val_loss:.4f}')
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

def train_test(paths_train, paths_test, espectrogram_name, sr, max_len, hop_length, win_length):
    if espectrogram_name == 'mel':
        espectrogram = mel_spectrogram
    elif espectrogram_name == 'mfcc':
        espectrogram = mfcc_combined
    elif espectrogram_name == 'cqt':
        espectrogram = CQT
    else: 
        raise ValueError("Espectrograma no soportado. Usa 'mel', 'mfcc' o 'cqt'")
    
    X_train, X_test = [], []
    for path in paths_train:
        x, _ = sf.read(path)
        X_train.append(espectrogram(x, sr=sr, max_len=max_len, hop_length=hop_length, win_length=win_length))

    for path in paths_test:
        x, _ = sf.read(path)
        X_test.append(espectrogram(x, sr=sr, max_len=max_len, hop_length=hop_length, win_length=win_length))

    return X_train, X_test