import torch
from torch import nn
import torch.nn.functional as F
from maskrcnn_benchmark.modeling.make_layers import group_norm


class GCNwithNonlocal(nn.Module):
    def __init__(self, plane):
        super(GCNwithNonlocal, self).__init__()
        inter_plane = plane // 2
        self.node_k = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_v = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_q = nn.Conv2d(plane, inter_plane, kernel_size=1)

        self.conv_wg = nn.Conv1d(inter_plane, inter_plane, kernel_size=1, bias=False)
        self.bn_wg = group_norm(inter_plane)
        self.softmax = nn.Softmax(dim=2)

        self.out = nn.Sequential(nn.Conv2d(inter_plane, plane, kernel_size=1),
                                 group_norm(plane))

    def forward(self, x):
        # b, c, h, w = x.size()
        node_k = self.node_k(x)
        node_v = self.node_v(x)
        node_q = self.node_q(x)
        b,c,h,w = node_k.size()
        node_k = node_k.view(b, c, -1).permute(0, 2, 1)
        node_q = node_q.view(b, c, -1)
        node_v = node_v.view(b, c, -1).permute(0, 2, 1)
        # A = k * q
        # AV = k * q * v
        # AVW = k *(q *v) * w
        AV = torch.bmm(node_q,node_v)
        AV = self.softmax(AV)
        AV = torch.bmm(node_k, AV)
        # node_v = node_v.permute()
        AV = AV.transpose(1, 2).contiguous()
        AVW = self.conv_wg(AV)
        AVW = self.bn_wg(AVW)
        AVW = AVW.view(b, c, h, -1)
        out = self.out(AVW) + x
        return out


class GloReLocalModule(nn.Module):
    def __init__(self, planes, ratio=4):
        super(GloReLocalModule, self).__init__()

        self.phi = nn.Conv2d(planes, planes // ratio * 2, kernel_size=1, bias=False)
        self.bn_phi = group_norm(planes // ratio * 2)
        self.theta = nn.Conv2d(planes, planes // ratio, kernel_size=1, bias=False)
        self.bn_theta = group_norm(planes // ratio)

        #  Interaction Space
        #  Adjacency Matrix: (-)A_g
        self.conv_adj = nn.Conv1d(planes // ratio, planes // ratio, kernel_size=1, bias=False)
        self.bn_adj = group_norm(planes // ratio)

        #  State Update Function: W_g
        self.conv_wg = nn.Conv1d(planes // ratio * 2, planes // ratio * 2, kernel_size=1, bias=False)
        self.bn_wg = group_norm(planes // ratio * 2)

        #  last fc
        self.conv3 = nn.Conv2d(planes // ratio * 2, planes, kernel_size=1, bias=False)
        self.bn3 = group_norm(planes)

        self.local = nn.Sequential(
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            group_norm(planes),
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            group_norm(planes))

        self.gcn_local_attention = GCNwithNonlocal(planes)

        self.sigmoid_spatial = nn.Sigmoid()

        self.final = nn.Sequential(nn.Conv2d(planes * 2, planes, kernel_size=1, bias=False),
                                   group_norm(planes))

        self.relu = nn.ReLU(inplace=True)

    def to_matrix(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, -1)
        return x

    def forward(self, x):
        # # # # Local # # # #
        feature = x
        local = self.local(feature)
        local = self.gcn_local_attention(local)
        local_feat = F.interpolate(local, size=x.size()[2:], mode='bilinear', align_corners=True)
        # local = self.sigmoid_spatial(local)
        # local_feat = x * local + x

        # # # # Projection Space # # # #
        x_sqz, b = x, x

        x_sqz = self.phi(x_sqz)
        x_sqz = self.bn_phi(x_sqz)
        x_sqz = self.relu(x_sqz)
        x_sqz = self.to_matrix(x_sqz)

        b = self.theta(b)
        b = self.bn_theta(b)
        b = self.relu(b)
        b = self.to_matrix(b)

        # Project
        z_idt = torch.matmul(x_sqz, b.transpose(1, 2))

        # # # # Interaction Space # # # #
        z = z_idt.transpose(1, 2).contiguous()

        z = self.conv_adj(z)
        z = self.bn_adj(z)
        z = self.relu(z)

        z = z.transpose(1, 2).contiguous()
        # Laplacian smoothing: (I - A_g)Z => Z - A_gZ
        z += z_idt

        z = self.conv_wg(z)
        z = self.bn_wg(z)
        z = self.relu(z)

        # # # # Re-projection Space # # # #
        # Re-project
        y = torch.matmul(z, b)

        n, _, h, w = x.size()
        y = y.view(n, -1, h, w)

        y = self.conv3(y)
        y = self.bn3(y)
        y = self.relu(y)

        g_out = x+y

        out = self.relu(self.final(torch.cat((local_feat, g_out), 1)))

        return out
