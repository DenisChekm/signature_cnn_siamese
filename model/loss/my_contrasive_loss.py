import torch


class MyContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=1.0):  # try change "1.0" to "2.0"
        super(MyContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, euclidean_distance, label):
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

    def get_margin(self):
        return self.margin
