import torch

# Configuración del dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")
torch.cuda.manual_seed(28)
torch.cuda.set_per_process_memory_fraction(0.5)

# Funciones auxiliares
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