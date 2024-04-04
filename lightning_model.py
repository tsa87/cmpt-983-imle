import lightning as L
from torch import optim

from emd import earth_mover_distance
from model import PointAE


class LitAE(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.autoencoder = PointAE(config)

    def _step(self, batch, batch_idx):
        pts = batch["points"]
        pts_enc = batch["points_encoded"]

        recon_pts = self.autoencoder(pts_enc)

        emd_loss = earth_mover_distance(recon_pts, pts)
        emd_loss = emd_loss.mean()

        return emd_loss

    def training_step(self, batch, batch_idx):
        emd_loss = self._step(batch, batch_idx)
        self.log("train_loss", emd_loss)
        return emd_loss

    def validation_step(self, batch, batch_idx):
        emd_loss = self._step(batch, batch_idx)
        self.log("val_loss", emd_loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=5e-4)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9995)
        return [optimizer], [scheduler]
