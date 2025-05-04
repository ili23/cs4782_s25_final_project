import lightning as L
import torch

class PatchTSTTrainer(L.LightningModule):
    def __init__(self, model, output_dir, lr=1e-3, lr_step=0.1):
        super().__init__()
        self.model = model
        self.output_dir = output_dir
        self.best_val_loss = float('inf')
        self.loss = torch.nn.MSELoss()
        self.lr_step = lr_step
        self.lr = lr

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.train_loss = 0
        x, y = batch[0], batch[1]
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.train_loss += loss.item()
        self.log('train_loss', loss)
        return loss
    
    def on_train_epoch_end(self):
        self.log('total_train_loss', self.train_loss)

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.val_loss = 0
        x, y = batch[0], batch[1]
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.val_loss += loss.item()
        self.log('val_loss', loss)
        return loss
    
    def on_validation_epoch_end(self):
        self.log('total_val_loss', self.val_loss)
        if self.val_loss < self.best_val_loss:
            self.best_val_loss = self.val_loss
            self.log('best_val_loss', self.val_loss)
            # Save the model checkpoint
            torch.save(self.model.state_dict(), self.output_dir + 'best_model.pt')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.lr_step)
        return [optimizer], [scheduler]