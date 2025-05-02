import lightning as L
import torch

class PatchTSTTrainer(L.LightningModule):
    def __init__(self, model, output_dir):
        super().__init__()
        self.model = model
        self.output_dir = output_dir
        self.best_val_loss = float('inf')
        self.loss = torch.nn.MSELoss()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.train_loss = 0
        x, y = batch
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
        x, y = batch
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]