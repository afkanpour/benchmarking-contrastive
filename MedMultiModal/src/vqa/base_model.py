"""
MEVF pipeline [1] with Bilinear Attention Network (BAN) for fusion [2].

References
----------
[1] Xuan B. Nguyen, URL: https://github.com/aioz-ai/MICCAI19-MedVQA/tree/master
[2] Jin-Hwa Kim, URL: https://github.com/jnhwkim/ban-vqa
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from lightning_utilities.core.rank_zero import rank_zero_info
from omegaconf import DictConfig
from open_clip import create_model
from torch import nn
from vqa.attention import BiAttention
from vqa.auto_encoder import AutoEncoderModel
from vqa.bc import BCNet
from vqa.classifier import SimpleClassifier
from vqa.fc import FCNet


# Create BAN model
class BANModel(nn.Module):
    """Bilinear Attention Network (BAN) [1].

    References
    ----------
    [1] Kim, Jin-Hwa, Jaehyun Jun, and Byoung-Tak Zhang. "Bilinear attention networks."
        Advances in neural information processing systems 31 (2018).
        URL: https://github.com/jnhwkim/ban-vqa
    """

    def __init__(
        self,
        cfg: DictConfig,
        image_text_model: nn.Module,
        visual_autoencoder: Optional[AutoEncoderModel],
        biattention_fusion_model: BiAttention,
        bilinear_connect_net: List[BCNet],
        q_prj: List[FCNet],
        classifier: SimpleClassifier,
    ) -> None:
        """Initialize the module."""
        super(BANModel, self).__init__()
        self.cfg = cfg
        self.glimpse = cfg.fusion.gamma

        self.model = image_text_model
        self.fusion_module = biattention_fusion_model
        self.b_net = nn.ModuleList(bilinear_connect_net)
        self.q_prj = nn.ModuleList(q_prj)
        self.classifier = classifier
        if cfg.autoencoder.enabled:
            assert isinstance(visual_autoencoder, AutoEncoderModel)
            self.visual_autoencoder: AutoEncoderModel = visual_autoencoder
            self.convert = nn.Linear(16384, cfg.autoencoder.feat_dim)

    def forward(
        self, batch: Dict[str, Any]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Encode image and text and fuse the embeddings.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch of data examples.
            The following keys must exist in `batch`:
                `"text"` : torch.Tensor
                    Tokenized text of size (batch, num_tokens).
                `"rgb"` : torch.Tensor
                    Image data of size (batch, num_images, height, width).

        Notes
        -----
        Returns logits, not probabilities.
        The logits should later be inputted into a classifier.
        """
        # get visual feature and embed texts
        visual_input = batch["rgb"]
        image_features = self.model.encode_image(
            visual_input[0] if self.cfg.autoencoder.enabled else visual_input
        )
        image_features = image_features.unsqueeze(1)

        # get DAE features
        if self.cfg.autoencoder.enabled:
            encoder = self.visual_autoencoder.forward_pass(visual_input[1])
            decoder = self.visual_autoencoder.reconstruct_pass(encoder)
            ae_v_emb = encoder.view(encoder.shape[0], -1)
            ae_v_emb = self.convert(ae_v_emb).unsqueeze(1)
            # concatenate encoder and autoencoder embeddings
            image_features = torch.cat((image_features, ae_v_emb), 2)

        # get lextual feature - [batch, q_len, q_dim]
        text_features = self.model.encode_text(batch["text"])

        # attention
        b_emb: List[torch.Tensor] = [0] * self.glimpse
        att, logits = self.fusion_module.forward_all(
            image_features, text_features
        )  # batch x glimpse x v_num_objs x q_num_objs
        for g in range(self.glimpse):
            b_emb[g] = self.b_net[g].forward_with_weights(
                image_features, text_features, att[:, g, :, :]
            )  # b x l x h
            _, _ = logits[:, g, :, :].max(2)
            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + text_features

        if self.cfg.autoencoder.enabled:
            return q_emb.sum(1), decoder
        return q_emb.sum(1)

    def classify(self, input_feats: torch.Tensor) -> torch.Tensor:
        """Classify fused embedding."""
        return self.classifier(input_feats)


# Build BAN model
def build_ban(cfg: DictConfig) -> BANModel:
    """Build the VQA pipeline with BAN fusion and optional MEVF."""
    # clip model (from openclip)
    model_kwargs = {}
    if "coca" in cfg.model_name and cfg.caption_encoder == "ViT-B-16":
        model_kwargs["caption_encoder"] = cfg.caption_encoder
    if cfg.filip:
        model_kwargs["filip"] = cfg.filip
    else:
        model_kwargs["token_level_embedding"] = True

    openclip_model = create_model(
        cfg.model_name,
        pretrained=cfg.pretrained,
        cache_dir=cfg.get("cache_dir", None),
        output_dict=True,
        **model_kwargs,
    )

    # auto-encoder
    visual_embedding_dim = openclip_model.visual.output_dim
    if cfg.autoencoder.enabled:
        visual_embedding_dim += cfg.autoencoder.feat_dim

    visual_autoencoder = None
    if cfg.autoencoder.enabled:
        visual_autoencoder = AutoEncoderModel()
        rank_zero_info(f"Loading initial weights DAE from: {cfg.autoencoder.file_path}")
        visual_autoencoder.load_state_dict(torch.load(cfg.autoencoder.file_path))

    # attention network
    num_hidden_dims = cfg.fusion.num_hid
    gamma = cfg.fusion.gamma
    biattention_fusion = BiAttention(
        visual_embedding_dim, num_hidden_dims, num_hidden_dims, gamma
    )

    # BAN residual network
    bilinear_connect_net: List[nn.Module] = []
    q_prj: List[nn.Module] = []
    for _ in range(gamma):
        bilinear_connect_net.append(
            BCNet(visual_embedding_dim, num_hidden_dims, num_hidden_dims, None, k=1)
        )
        q_prj.append(FCNet([num_hidden_dims, num_hidden_dims], "", 0.2))

    # classifier
    classifier = SimpleClassifier(
        num_hidden_dims, num_hidden_dims * 2, cfg.classifier.num_classes, cfg.classifier
    )

    return BANModel(
        cfg,
        openclip_model,
        visual_autoencoder,
        biattention_fusion,
        bilinear_connect_net,
        q_prj,
        classifier,
    )
