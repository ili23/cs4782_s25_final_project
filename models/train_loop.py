import lightning as L
import torch

class PatchTSTTrainer(L.LightningModule):
    def __init__(self, model, output_dir, lr=1e-3, lr_step=0.7):
        super().__init__()
        self.model = model
        self.output_dir = output_dir
        self.best_val_loss = float('inf')
        self.loss = torch.nn.MSELoss()
        self.mae_loss = torch.nn.L1Loss()
        self.pred_f = -1
        self.lr_step = lr_step
        self.lr = lr

    def forward(self, x):
        return self.model(x)
    
    def get_loss(self, y_hat, y):
        return self.loss(y_hat, y)
    
    def get_mae_loss(self, y_hat, y):
        return self.mae_loss(y_hat, y)
    
    def get_target_loss(self, y_hat, y):
        y = y[:, :, self.pred_f:]
        y_hat = y_hat[:, :, self.pred_f:]
        return self.loss(y_hat, y)
    
    def get_target_loss(self, y_hat, y):
        y = y[:, :, self.pred_f:]
        y_hat = y_hat[:, :, self.pred_f:]
        return self.mae_loss(y_hat, y)
    
    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.train_loss = []
        x, y = batch[0], batch[1]
        # print("target shape: ", y.shape)
        y_hat = self.forward(x)
        loss = self.get_loss(y_hat, y)
        self.train_loss.append(loss.item())
        self.log('train_loss', loss)
        return loss
    
    def on_train_epoch_end(self):
        train_loss = torch.mean(torch.tensor(self.train_loss))
        self.log('total_train_loss', train_loss)
        print("Train epoch complete, loss: ", train_loss)

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.val_loss = []
        x, y = batch[0], batch[1]
        y_hat = self.forward(x)
        loss = self.get_loss(y_hat, y)
        self.val_loss.append(loss.item())
        self.log('val_loss', loss)
        return loss
    
    def on_validation_epoch_end(self):
        val_loss = torch.mean(torch.tensor(self.val_loss))
        self.log('total_val_loss', val_loss)
        print("Validation epoch complete, loss: ", val_loss)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.log('best_val_loss', val_loss)
            # Save the model checkpoint
            torch.save(self.model.state_dict(), self.output_dir + 'best_model.pt')

    def load_best_model(self):
        # Load the best model checkpoint
        self.model.load_state_dict(torch.load(self.output_dir + 'best_model.pt'))
        self.model.eval()
        return self.model

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.model = self.load_best_model()
            self.test_loss = []
            self.test_mae_loss = []
        x, y = batch[0], batch[1]
        y_hat = self.forward(x)
        loss = self.get_loss(y_hat, y)
        mae_loss = self.get_mae_loss(y_hat, y)
        self.test_loss.append(loss.item())
        self.test_mae_loss.append(mae_loss.item())
        self.log('test_loss', loss)
        self.log('test_mae_loss', mae_loss)
        return loss

    def on_test_epoch_end(self):
        test_loss = torch.mean(torch.tensor(self.test_loss))
        test_mae_loss = torch.mean(torch.tensor(self.test_mae_loss))
        self.log('total_test_loss', test_loss)
        self.log('total_test_mae_loss', test_mae_loss)
        print("Test epoch complete, loss: ", test_loss)
        print("Test epoch complete, mae loss: ", test_mae_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=self.lr_step)
        return [optimizer], [scheduler]
        # return optimizer