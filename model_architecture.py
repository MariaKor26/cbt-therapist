import torch
import torch.nn as nn


class PatternAttentionRobertaModel(nn.Module):
    def __init__(self, backbone, num_patterns, hidden_size=768):
        super(PatternAttentionRobertaModel, self).__init__()
        self.backbone = backbone  # DeBERTa-v3-small loaded externally
        self.num_patterns = num_patterns
        self.dropout = nn.Dropout(p=0.4)

        self.pattern_head = nn.Linear(hidden_size, num_patterns)


        self.pattern_embeddings = nn.Embedding(num_patterns, hidden_size)

        # ONE cross-attention layer only
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4, batch_first=True)

        self.norm = nn.LayerNorm(hidden_size)
        self.act = nn.GELU()




    def forward(self, input_ids, attention_mask, pattern_labels=None, use_gold_patterns=False):


        # === 2. Run DeBERTa Encoder
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = output.last_hidden_state[:, 0, :]  # [CLS] token output

        # === 3. Predict Patterns (first head)
        pattern_logits = self.pattern_head(cls_output)

        if use_gold_patterns and pattern_labels is not None:
            pattern_mask = pattern_labels.float()
        else:
            pattern_mask = torch.sigmoid(pattern_logits).detach()


        # === 4. Prepare Pattern Embeddings
        pattern_embed = self.pattern_embeddings(torch.arange(self.num_patterns).to(input_ids.device))
        pattern_embed = pattern_embed.unsqueeze(0).expand(input_ids.size(0), -1, -1)

        # === 5. Cross Attention
        query = cls_output.unsqueeze(1)
        attn_output, attn_weights = self.cross_attention(query, pattern_embed, pattern_embed)
        attn_output = self.norm(self.act(attn_output))
        attn_output = self.dropout(attn_output)

        pattern_logits = self.pattern_head(attn_output.squeeze(1))  # [B, num_patterns]

        return pattern_logits, attn_weights