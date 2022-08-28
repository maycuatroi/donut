from pytorch_lightning.callbacks import Callback


class DonutCheckPointCallBack(Callback):
    def __init__(self, save_path: str):
        self.save_path = save_path
        super().__init__()

    def on_train_epoch_end(self, trainer, pl_module):
        print(f"Save model, epoch {trainer.current_epoch}")
        pl_module.model.save_pretrained(self.save_path)

    def on_train_end(self, trainer, pl_module):
        print("Save model checkpoints after training")
        pl_module.model.save_pretrained(self.save_path)
        pl_module.model.processor.save_pretrained(self.save_path)
