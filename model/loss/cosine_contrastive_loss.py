import torch
from torch.nn.functional import cosine_similarity, normalize


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    # def forward(self, cosine_similarity, label):
    #     return torch.mean((1 - label) * torch.pow(cosine_similarity, 2) + label * torch.pow(torch.clamp(self.margin - cosine_similarity, min=0.0), 2))

    def forward(self, embedding1, embedding2, label):
        similarity = cosine_similarity(normalize(embedding1), normalize(embedding2))
        return torch.mean((1 - label) * torch.pow(similarity, 2) + label * torch.pow(torch.clamp(self.margin - similarity, min=0.0), 2))

    # def forward(self, eucl_dist, label):
    #     return torch.mean((1 - label) * torch.pow(eucl_dist, 2) + label * torch.pow(torch.clamp(self.margin - eucl_dist, min=0.0), 2))

    def get_margin(self):
        return self.margin
