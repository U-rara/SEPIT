import torch
import torch.nn.functional as F
from torch import nn, Tensor


@torch.jit.script
def gaussian(x, mean, std):
    pi = torch.pi
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.K = K
        self.means = nn.Parameter(torch.zeros(1, K))
        self.stds = nn.Parameter(torch.zeros(1, K))
        nn.init.uniform_(self.means, 0, 3)
        nn.init.uniform_(self.stds, 0, 3)

    def forward(self, x):
        x = x.unsqueeze(-1).expand(-1, -1, -1, self.K)
        mean = self.means.float().view(-1)
        std = self.stds.float().view(-1).abs() + 1e-3
        return gaussian(x.float(), mean, std).type_as(self.means)


class NonLinear(nn.Module):
    def __init__(self, input, output_size, hidden=None):
        super(NonLinear, self).__init__()

        if hidden is None:
            hidden = input
        self.layer1 = nn.Linear(input, hidden)
        self.layer2 = nn.Linear(hidden, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = F.gelu(x)
        x = self.layer2(x)
        return x


class StructureBias(nn.Module):
    def __init__(self, num_attention_heads, hidden_size, num_gaussian_basis_kernel=128):
        super(StructureBias, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.num_gaussian_basis_kernel = num_gaussian_basis_kernel
        self.hidden_size = hidden_size

        self.gbf = GaussianLayer(self.num_gaussian_basis_kernel)
        self.gbf_proj = NonLinear(
            self.num_gaussian_basis_kernel, self.num_attention_heads
        )

        if self.num_gaussian_basis_kernel != self.hidden_size:
            self.edge_proj = nn.Linear(self.num_gaussian_basis_kernel, self.hidden_size)
        else:
            self.edge_proj = None

    def forward(self, attention_mask, node_position):
        node_position_mask = torch.all(node_position == 0, dim=-1)

        # pos shape: [batch_size, max_length, 3]
        bsz, n_node, _ = node_position.shape
        delta_pos = node_position.unsqueeze(1) - node_position.unsqueeze(2)
        dist = delta_pos.norm(dim=-1).view(-1, n_node, n_node)

        edge_feature = self.gbf(dist)
        gbf_result = self.gbf_proj(edge_feature)
        graph_attn_bias = gbf_result

        graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
        attention_mask = 1 - attention_mask

        graph_attn_bias.masked_fill_(
            attention_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
        )

        edge_feature = edge_feature.masked_fill(
            attention_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool), 0.0
        )

        sum_edge_features = edge_feature.sum(dim=-2)
        merge_edge_features = self.edge_proj(sum_edge_features)

        graph_attn_bias = graph_attn_bias.masked_fill(
            node_position_mask.unsqueeze(1).unsqueeze(2), 0.0
        )
        merge_edge_features = merge_edge_features.masked_fill(
            node_position_mask.unsqueeze(-1), 0.0
        )

        return graph_attn_bias, merge_edge_features


class PositionHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.scaling = (embed_dim // num_heads) ** -0.5
        self.force_proj1 = nn.Linear(embed_dim, 1)
        self.force_proj2 = nn.Linear(embed_dim, 1)
        self.force_proj3 = nn.Linear(embed_dim, 1)
        self.dropout_module = nn.Dropout(0.1)

    def forward(
        self,
        query: Tensor,
        attention_bias: Tensor,
        node_position: Tensor,
    ) -> Tensor:
        bsz, n_node, _ = node_position.shape
        delta_pos = node_position.unsqueeze(1) - node_position.unsqueeze(2)
        dist = delta_pos.norm(dim=-1).view(-1, n_node, n_node)
        delta_pos /= dist.unsqueeze(-1) + 1e-5

        query = query.contiguous().transpose(0, 1)
        q = (
            self.q_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
            * self.scaling
        )
        k = self.k_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
        v = self.v_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
        attn = q @ k.transpose(-1, -2)  # [bsz, head, n, n]
        attn_probs_float = torch.softmax(
            attn.view(-1, n_node, n_node)
            + attention_bias.contiguous().view(-1, n_node, n_node),
            dim=-1,
        )
        attn_probs = attn_probs_float.type_as(attn)
        attn_probs = self.dropout_module(attn_probs).view(
            bsz, self.num_heads, n_node, n_node
        )
        rot_attn_probs = attn_probs.unsqueeze(-1) * delta_pos.unsqueeze(1).type_as(
            attn_probs
        )  # [bsz, head, n, n, 3]
        rot_attn_probs = rot_attn_probs.permute(0, 1, 4, 2, 3)
        x = rot_attn_probs @ v.unsqueeze(2)  # [bsz, head , 3, n, d]
        x = x.permute(0, 3, 2, 1, 4).contiguous().view(bsz, n_node, 3, -1)
        f1 = self.force_proj1(x[:, :, 0, :]).view(bsz, n_node, 1)
        f2 = self.force_proj2(x[:, :, 1, :]).view(bsz, n_node, 1)
        f3 = self.force_proj3(x[:, :, 2, :]).view(bsz, n_node, 1)
        cur_force = torch.cat([f1, f2, f3], dim=-1).float()
        return cur_force
