from models.BaseModel import GeneralModel
import torch.nn as nn

class NewModel(GeneralModel):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size']          # 可选，方便日志打印

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size      # ⭐ 先赋值
        self._define_params()
        self.apply(self.init_weights)

    def _define_params(self):
        self.user_emb = nn.Embedding(self.user_num, self.emb_size)
        self.item_emb = nn.Embedding(self.item_num, self.emb_size)

    def forward(self, feed_dict):
        user_id = feed_dict['user_id']              # [B]
        item_id = feed_dict['item_id']              # [B, 1+neg]
        u = self.user_emb(user_id).unsqueeze(1)     # [B, 1, dim]
        i = self.item_emb(item_id)                  # [B, 1+neg, dim]
        pred = (u * i).sum(dim=-1)                  # [B, 1+neg]
        return {'prediction': pred}