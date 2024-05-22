# 3p
import torch
import torch.nn as nn
import roma
from diffusion_net.layers import DiffusionNet


class PrismDecoder(torch.nn.Module):
    def __init__(self, C_in =1024, C_out=512, C_width=256, N_block=4, dropout=False, num_eigenbasis=128):
        super().__init__()

        self.diffusion_net = DiffusionNet(C_in=C_in , C_out=C_out, C_width=C_width,
                                          N_block=N_block, dropout=dropout, num_eigenbasis=num_eigenbasis)

        self.mlp_refine = nn.Sequential(
            nn.Linear(C_out, C_out),
            nn.ReLU(),
            nn.Linear(C_out, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 12),
        )

    def forward(self, batch_data):
        # original prism
        # todo 
        verts = batch_data.pos.reshape(-1, 3)
        faces = batch_data.faces.t()
        prism_base = verts[faces]  # (n_faces, 3, 3)
        # bs, _, _ = batch_data.pos.shape
        bs = 1

        # forward through diffusion net
        batch_data = self.diffusion_net(batch_data)  # (bs, n_verts, dim)

        # features per face
        x = batch_data.features
        x_gather = x.unsqueeze(-1).expand(-1, -1, -1, 3)
        faces_gather = faces.unsqueeze(0).unsqueeze(2).expand(-1, -1, x.shape[-1], -1)
        xf = torch.gather(x_gather, 1, faces_gather)
        features = torch.mean(xf, dim=-1)  # (bs, n_faces, dim)

        # refine features with mlp
        features = self.mlp_refine(features)  # (bs, n_faces, 12)

        # get the translation and rotation
        rotations = features[:, :, :9].reshape(-1, 3, 3)
        rotations = roma.special_procrustes(rotations)  # (n_faces, 3, 3)
        translations = features[:, :, 9:].reshape(-1, 3)  # (n_faces, 3)

        # transform the prism
        transformed_prism = (prism_base @ rotations) + translations[:, None]

        # prism to vertices
        features = self.prism_to_vertices(transformed_prism, faces, verts)

        batch_data.features = features.reshape(bs, -1, 3)
        batch_data.transformed_prism = transformed_prism
        batch_data.rotations = rotations
        return batch_data

    def prism_to_vertices(self, prism, faces, verts):
        # initialize the transformed features tensor
        N = verts.shape[0]
        d = prism.shape[-1]
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        features = torch.zeros((N, d), device=device)

        # scatter the features in K onto L using the indices in F
        features.scatter_add_(0, faces[:, :, None].repeat(1, 1, d).reshape(-1, d), prism.reshape(-1, d))

        # divide each row in the transformed features tensor by the number of faces that the corresponding vertex appears in
        num_faces_per_vertex = torch.zeros(N, dtype=torch.float32, device=device)
        num_faces_per_vertex.index_add_(0, faces.reshape(-1), torch.ones(faces.shape[0] * 3, device=device))
        features /= num_faces_per_vertex.unsqueeze(1).clamp(min=1)

        return features
