from utils import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def fit(checkpoint, train_loader, val_loader, device,epochs, writers=None, mode="FC"):
    """Full training loop"""

    
    net, optimizer = checkpoint.model, checkpoint.optimizer
    min_loss = float('inf')
    iteration = 1

    def train_epoch():
        """
        Returns:
            The epoch loss
        """
        nonlocal iteration
        epoch_loss = 0.
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}', dynamic_ncols=True,  position=0, leave=True)  # progress bar
        net.train()
        for (batch_x,_) in pbar:
            optimizer.zero_grad()
            iteration += 1

            # (batch_size, w*h)
            if mode =="FC":
                batch_x = batch_x.view((batch_x.shape[0], -1))

            batch_x = batch_x.to(device)

            batch_x_hat, mu, log_var = net(batch_x)
            loss = loss_function(batch_x_hat, batch_x, mu, log_var, mode)
            pbar.set_postfix(loss=f'{loss.item():.4e}')
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            iteration += 1

        epoch_loss /= len(train_loader)
        return epoch_loss
    def evaluate_epoch(loader,epoch, role='Val'):
        """
        Args:
            loader (torch.utils.data.DataLoader): either the train of validation loader
            role (str): either 'Val' or 'Train'
        Returns:
            Tuple containing mean loss and accuracy
        """
        net.eval()
        mean_loss = 0.
        im, _ = next(iter(loader))
        im = im.to(device)
        if role =="Train":
            create_grid(net, im, writers['images'], epoch, mode)
        with torch.no_grad():
            for step, (batch_x, _) in enumerate(loader):
                if mode =="FC":
                    batch_x = batch_x.view((batch_x.shape[0], -1))

                batch_x = batch_x.to(device)
                batch_x_hat, mu, log_var = net(batch_x)
                loss = loss_function(batch_x_hat, batch_x, mu, log_var)

                mean_loss += loss.item()
                
        
        return mean_loss / len(loader)

    begin_epoch = checkpoint.epoch

    for epoch in range(begin_epoch, epochs+1):

        train_epoch()
        loss_train = evaluate_epoch(train_loader,epoch, 'Train')
        loss_test =  evaluate_epoch(val_loader,epoch, 'Val')
        print(f"Epoch {epoch}/{epochs}, Train Loss: {loss_train:.4e}, Test Loss: {loss_test:.4f}")

        writers['loss'].add_scalars('loss', {'train': loss_train, 'test': loss_test}, epoch)
        checkpoint.epoch += 1
        if loss_test < min_loss:
            min_loss = loss_test
            checkpoint.save('_best')
        checkpoint.save()

    print("\nFinished.")
    print(f"Best validation loss: {min_loss:.4e}")
    
def test(model, test_loader, device, writer):
    im, _ = next(iter(test_loader))
    im = im.to(device)
    create_grid(model, im, writer=writer, epoch=0)
    
