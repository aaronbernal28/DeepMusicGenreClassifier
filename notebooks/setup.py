import torch
from sklearn.model_selection import train_test_split

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

    print("Entrenamiento completado!")
    return train_losses, val_losses