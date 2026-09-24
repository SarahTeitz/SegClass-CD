# this section deals with GeM pooling, modality weighting and classification head

# imports:
import torch
import torch.nn.functional as F
import torch.nn as nn

# masked GeM:
class MaskedGeMPool(nn.Module):
    def __init__(self, p=3.0, eps=1e-6, num_channels=1): # need to add in the channels
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p))) # turn this off while switching to channels based
        # self.p = nn.Parameter(torch.full((num_channels,), float(p))) # [C]
        self.eps = eps

    def forward(self, feats, mask):  # feats: [B,C,D,H,W], mask: [B,D,H,W]
        mask = mask.clamp(0, 1)

        denom = mask.sum(dim=(2,3,4), keepdim=True).clamp_min(self.eps)
        p = torch.clamp(self.p, 1e-3, 10.0)
        # p = p.view(1, -1, 1, 1, 1)  # -> [1,C,1,1,1] for broadcasting
        x = F.relu(feats).clamp_min(1e-6).pow(p)
        x = (x * mask).sum(dim=(2,3,4), keepdim=True) / denom
        x = x.clamp_min(1e-6).pow(1.0 / p)
        return x.squeeze(-1).squeeze(-1).squeeze(-1)  # [B,C]

class multiple_featuremaps_GEM_classification(nn.Module): # for single-contrast model - used to downsample the segmentation masks
    def __init__(self, in_channels, label_num):
        super(multiple_featuremaps_GEM_classification, self).__init__()
        self.GEM_head = GeM_multiple_feature_maps_classifier(in_channels, num_classes=label_num)
    def forward(self, featuremap1, featuremap2, featuremap3, featuremap4, featuremap5, seg):
        downsampled_seg2 = F.max_pool3d(seg.float(), kernel_size=(1, 2, 2), stride=(1, 2, 2))
        downsampled_seg3 = F.max_pool3d(seg.float(), kernel_size=(1, 4, 4), stride=(1, 4, 4))
        downsampled_seg4 = F.max_pool3d(seg.float(), kernel_size=(2, 8, 8), stride=(2, 8, 8))
        downsampled_seg5 = F.max_pool3d(seg.float(), kernel_size=(4, 16, 16), stride=(4, 16, 16))
        classification = self.GEM_head(featuremap1, seg, featuremap2,downsampled_seg2, featuremap3, downsampled_seg3, featuremap4, downsampled_seg4, featuremap5, downsampled_seg5)
        return classification

class GeM_multiple_feature_maps_classifier(nn.Module): # Level GeM pooling + classification head - single contrast model
    def __init__(self, in_channels, num_classes=1, p_init=3.0):
        super(GeM_multiple_feature_maps_classifier, self).__init__()
        self.pool1 = MaskedGeMPool(p=p_init, num_channels=32)
        self.pool2 = MaskedGeMPool(p=p_init, num_channels=64)
        self.pool3 = MaskedGeMPool(p=p_init, num_channels=128)
        self.pool4 = MaskedGeMPool(p=p_init, num_channels=256)
        self.pool5 = MaskedGeMPool(p=p_init, num_channels=320)

        self.classification_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(in_channels, 256),
                            nn.ReLU(),
                            nn.Dropout(p=0.2),
                            nn.Linear(256, 1),
            )
            for _ in range(num_classes)
        ])

    def forward(self, featuremap1, mask1, featuremap2, mask2, featuremap3, mask3, featuremap4, mask4, featuremap5, mask5):
        z_1 = self.pool1(featuremap1, mask1)  # [B,C]
        z_2 = self.pool2(featuremap2, mask2)
        z_3 = self.pool3(featuremap3, mask3)
        z_4 = self.pool4(featuremap4, mask4)
        z_5 = self.pool5(featuremap5, mask5)
        z = torch.cat((z_1, z_2, z_3, z_4, z_5), dim=1)
        # classification = self.fc(z)
        classification = torch.cat([head(z) for head in self.classification_heads], dim=1)
        return classification #self.classification_heads(z)  # [B,num_classes]


class t1_t2_GEM_head(nn.Module): # used to downsample segmentation for the multi-contrast model
    def __init__(self, in_channels, label_num, modality_type):
        super(t1_t2_GEM_head, self).__init__()
        self.GEM_head = t1_t2_GEM_classifier(in_channels, num_classes=label_num, modality_type=modality_type)
    def forward(self, seg, t1_1, t1_2, t1_3,t1_4, t1_5, t2_1, t2_2, t2_3, t2_4, t2_5):
        downsampled_seg2 = F.max_pool3d(seg.float(), kernel_size=(1, 2, 2), stride=(1, 2, 2))
        downsampled_seg3 = F.max_pool3d(seg.float(), kernel_size=(1, 4, 4), stride=(1, 4, 4))
        downsampled_seg4 = F.max_pool3d(seg.float(), kernel_size=(2, 8, 8), stride=(2, 8, 8))
        downsampled_seg5 = F.max_pool3d(seg.float(), kernel_size=(4, 16, 16), stride=(4, 16, 16))
        classification = self.GEM_head(seg, downsampled_seg2,downsampled_seg3,downsampled_seg4,downsampled_seg5,
                                       t1_1, t1_2, t1_3, t1_4, t1_5, t2_1, t2_2, t2_3, t2_4, t2_5)
        return classification


class t1_t2_GEM_classifier(nn.Module): # for multi-contrast model
    def __init__(self, in_channels, num_classes=1, p_init=3.0, modality_type = ''):
        super(t1_t2_GEM_classifier, self).__init__()
        self.modality_type = modality_type
        self.t1pool1 = MaskedGeMPool(p=p_init)
        self.t1pool2 = MaskedGeMPool(p=p_init)
        self.t1pool3 = MaskedGeMPool(p=p_init)
        self.t1pool4 = MaskedGeMPool(p=p_init)
        self.t1pool5 = MaskedGeMPool(p=p_init)
        self.t2pool1 = MaskedGeMPool(p=p_init)
        self.t2pool2 = MaskedGeMPool(p=p_init)
        self.t2pool3 = MaskedGeMPool(p=p_init)
        self.t2pool4 = MaskedGeMPool(p=p_init)
        self.t2pool5 = MaskedGeMPool(p=p_init)

        if modality_type == 'linear':
            self.a = nn.Parameter(torch.rand(num_classes))

        elif modality_type == 'learned':
            self.learned_classification = LearnedModalityWeightedClassifier(in_channels, 256, num_classes) #learned modality

        elif modality_type == 'attention':
            self.modality_atten = TwoModalAttentionClassifier(input_dim= in_channels//2, num_classes=num_classes)
        # self.feature_atten = FeatureAttentionClassifier(input_dim= in_channels//2, num_classes=num_classes)

        if modality_type == "linear" or modality_type == "non":
            self.classification_heads = nn.ModuleList([
                  nn.Sequential(                # two layers
                      nn.Dropout(p=0.2),
                      nn.Linear(in_channels, 512),
                      nn.ReLU(),
                      nn.Dropout(p=0.2),
                      nn.Linear(512, 1),
                      )
                for _ in range(num_classes)
            ])

    def forward(self,  mask1, mask2, mask3,  mask4, mask5, t1_1, t1_2, t1_3, t1_4, t1_5, t2_1, t2_2, t2_3, t2_4, t2_5):  # feats from your ResNet stage
        # when I'll add the colon, I will need to pool per segmentation and then fc for each and then concatenate
        z_1 = self.t1pool1(t1_1, mask1)  # [B,C]
        z_2 = self.t1pool2(t1_2, mask2)
        z_3 = self.t1pool3(t1_3, mask3)
        z_4 = self.t1pool4(t1_4, mask4)
        z_5 = self.t1pool5(t1_5, mask5)
        z_6 = self.t2pool1(t2_1, mask1)  # [B,C]
        z_7 = self.t2pool2(t2_2, mask2)
        z_8 = self.t2pool3(t2_3, mask3)
        z_9 = self.t2pool4(t2_4, mask4)
        z_10 = self.t2pool5(t2_5, mask5)
        z_t1 = torch.cat((z_1, z_2, z_3, z_4, z_5), dim=1)
        z_t2 = torch.cat((z_6, z_7, z_8, z_9, z_10 ), dim=1)
        #-----------------
        if self.modality_type == 'non':
            z = torch.cat((z_1, z_2, z_3, z_4, z_5, z_6, z_7, z_8, z_9, z_10 ), dim=1) #- prior to modality weighting
            classification = torch.cat([head(z) for head in self.classification_heads], dim=1) #- prior to modality weighting
        #------------------
        elif self.modality_type == 'linear':
        # linear classification:
            outputs = []
            for i, head in enumerate(self.classification_heads):
                a_i = torch.sigmoid(self.a[i])
                z_i = torch.cat(((1 - a_i) * z_t1, a_i * z_t2),dim=1)
                outputs.append(head(z_i))
            classification = torch.cat(outputs, dim=1)
        #-----------------
        # Learned modality weighting:
        elif self.modality_type == 'learned':
            classification = self.learned_classification(z_t1, z_t2)
        #-----------------------------------------------
        # modality attention:
        else: #self.modality_type == 'attention':
            classification = self.modality_atten(z_t1, z_t2)

        return classification #self.classification_heads(z)  # [B,num_classes]


class LearnedModalityWeightedClassifier(nn.Module): # modality weighting num. 3
    def __init__(self, channels_num, d_model, num_classes):
        super().__init__()

        self.proj1 = nn.Linear(channels_num//2, d_model)
        self.proj2 = nn.Linear(channels_num//2, d_model)

        # attention score per modality
        self.modality_attn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, num_classes) # used to be 1
        )

        self.classification_heads = nn.ModuleList([
            nn.Linear(d_model, 1)
            for _ in range(num_classes)
        ])

    def forward(self, x1, x2):

        h1 = self.proj1(x1)  # [B, D]
        h2 = self.proj2(x2)  # [B, D]

        H = torch.stack([h1, h2], dim=1)  # [B, 2, D]

        scores = self.modality_attn(H)   # [B, 2, 1] -> [B,2,C] in newer version
        weights = torch.softmax(scores, dim=1)

        Z = torch.einsum("bmd,bmc->bcd", H, weights)  # [B, C, D]

        outputs = [
            head(Z[:, c, :])  # [B, D] -> [B, 1]
            for c, head in enumerate(self.classification_heads)
        ]
        return torch.cat(outputs, dim=1)

class TwoModalAttentionClassifier(nn.Module): # modality weighting num. 4
    def __init__(self, input_dim=800, embed_dim=128, num_heads=2, hidden_dim=256, num_classes=8, dropout = 0.3): #embed was 128
        super().__init__()

        self.num_classes=num_classes

        self.downsample1 = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim)
        )

        self.downsample2 = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim)
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )

        self.mlp = nn.Sequential(
            nn.Linear(2*embed_dim, num_classes) #shorten it to avoid overfitting
        )


    def forward(self, x1, x2):
        # x1, x2: [B, 800]

        z1 = self.downsample1(x1)  # [B, D]
        z2 = self.downsample2(x2)  # [B, D]

        z = torch.stack([z1, z2], dim=1)  # [B, 2, D]

        # logits: [B, 8]

        attended, attn_weights = self.attention(z, z, z)
        # attended: [B, 2, D]

        fused = attended.reshape(attended.size(0), -1)  # [B, 2D]

        logits = self.mlp(fused)  # [B, 8]

        return logits

