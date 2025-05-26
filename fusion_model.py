import torch
import torch.nn as nn
import torchvision.models as models
from transformers import ViTModel

class FusionModel(nn.Module):
    def __init__(self, num_classes=4, prompt_dim=128, wavelet_dim=4096):
        super(FusionModel, self).__init__()

        resnet = models.resnet50(weights=None)
        resnet.fc = nn.Identity()
        self.cnn = resnet
        self.cnn_out = 2048

        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        for param in self.vit.parameters():
            param.requires_grad = False
        self.vit_out = self.vit.config.hidden_size

        self.prompt_embed = nn.Embedding(num_classes, prompt_dim)

        total_features = self.cnn_out + self.vit_out + prompt_dim + wavelet_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(total_features),
            nn.Linear(total_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, wavelet_feats, labels):
        cnn_feats = self.cnn(images)
        vit_input = (images * 0.5) + 0.5
        vit_out = self.vit(pixel_values=vit_input).last_hidden_state[:, 0, :]
        prompt = self.prompt_embed(labels)
        wavelet_feats = (wavelet_feats - wavelet_feats.mean(dim=1, keepdim=True)) / (
            wavelet_feats.std(dim=1, keepdim=True) + 1e-5)
        fused = torch.cat([cnn_feats, vit_out, prompt, wavelet_feats], dim=1)
        return self.classifier(fused)
