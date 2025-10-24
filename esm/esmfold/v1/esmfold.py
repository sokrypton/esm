# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import typing as T
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from omegaconf import MISSING
from openfold.data.data_transforms import make_atom14_masks
from openfold.np import residue_constants
from openfold.utils.loss import compute_predicted_aligned_error, compute_tm
from torch import nn
from torch.nn import LayerNorm

import esm
from esm import Alphabet
from esm.esmfold.v1.categorical_mixture import categorical_lddt
from esm.esmfold.v1.trunk import FoldingTrunk, FoldingTrunkConfig
from esm.esmfold.v1.misc import (
    batch_encode_sequences,
    collate_dense_tensors,
    output_to_pdb,
)


@dataclass
class ESMFoldConfig:
    trunk: T.Any = field(default_factory=FoldingTrunkConfig)
    lddt_head_hid_dim: int = 128


class ESMFold(nn.Module):
    def __init__(self, esmfold_config=None, **kwargs):
        super().__init__()

        self.cfg = esmfold_config if esmfold_config else ESMFoldConfig(**kwargs)
        cfg = self.cfg

        self.distogram_bins = 64

        self.esm, self.esm_dict = esm.pretrained.esm2_t36_3B_UR50D()

        self.esm.requires_grad_(False)
        self.esm.half()

        self.esm_feats = self.esm.embed_dim
        self.esm_attns = self.esm.num_layers * self.esm.attention_heads
        self.register_buffer("af2_to_esm", ESMFold._af2_to_esm(self.esm_dict))
        self.esm_s_combine = nn.Parameter(torch.zeros(self.esm.num_layers + 1))

        c_s = cfg.trunk.sequence_state_dim
        c_z = cfg.trunk.pairwise_state_dim

        self.esm_s_mlp = nn.Sequential(
            LayerNorm(self.esm_feats),
            nn.Linear(self.esm_feats, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
        )

        # 0 is padding, N is unknown residues, N + 1 is mask.
        self.n_tokens_embed = residue_constants.restype_num + 3
        self.pad_idx = 0
        self.unk_idx = self.n_tokens_embed - 2
        self.mask_idx = self.n_tokens_embed - 1
        self.embedding = nn.Embedding(self.n_tokens_embed, c_s, padding_idx=0)

        self.trunk = FoldingTrunk(**cfg.trunk)

        self.distogram_head = nn.Linear(c_z, self.distogram_bins)
        self.ptm_head = nn.Linear(c_z, self.distogram_bins)
        self.lm_head = nn.Linear(c_s, self.n_tokens_embed)
        self.lddt_bins = 50
        self.lddt_head = nn.Sequential(
            nn.LayerNorm(cfg.trunk.structure_module.c_s),
            nn.Linear(cfg.trunk.structure_module.c_s, cfg.lddt_head_hid_dim),
            nn.Linear(cfg.lddt_head_hid_dim, cfg.lddt_head_hid_dim),
            nn.Linear(cfg.lddt_head_hid_dim, 37 * self.lddt_bins),
        )

    @staticmethod
    def _af2_to_esm(d: Alphabet):
        # Remember that t is shifted from residue_constants by 1 (0 is padding).
        esm_reorder = [d.padding_idx] + [
            d.get_idx(v) for v in residue_constants.restypes_with_x
        ]
        return torch.tensor(esm_reorder)

    def _af2_idx_to_esm_idx(self, aa, mask):
        aa = (aa + 1).masked_fill(mask != 1, 0)
        return self.af2_to_esm[aa]

    def _compute_language_model_representations(self,
        esmaa: torch.Tensor,
        return_contacts: bool = False
    ) -> torch.Tensor:
        """Adds bos/eos tokens for the language model, since the structure module doesn't use these."""
        batch_size = esmaa.size(0)

        bosi, eosi = self.esm_dict.cls_idx, self.esm_dict.eos_idx
        bos = esmaa.new_full((batch_size, 1), bosi)
        eos = esmaa.new_full((batch_size, 1), self.esm_dict.padding_idx)
        esmaa = torch.cat([bos, esmaa, eos], dim=1)
        # Use the first padding index as eos during inference.
        esmaa[range(batch_size), (esmaa != 1).sum(1)] = eosi

        res = self.esm(
            esmaa,
            repr_layers=range(self.esm.num_layers + 1),
            need_head_weights=False,
            return_contacts=return_contacts
        )
        esm_s = torch.stack(
            [v for _, v in sorted(res["representations"].items())], dim=2
        )
        esm_s = esm_s[:, 1:-1]  # B, L, nLayers, C
        return esm_s, res

    def _mask_inputs_to_esm(self, esmaa, pattern):
        new_esmaa = esmaa.clone()
        new_esmaa[pattern == 1] = self.esm_dict.mask_idx
        return new_esmaa

    def forward(
        self,
        aa: torch.Tensor,
        mask: T.Optional[torch.Tensor] = None,
        residx: T.Optional[torch.Tensor] = None,
        masking_pattern: T.Optional[torch.Tensor] = None,
        num_recycles: T.Optional[int] = None,
        mask_rate: float = 0.0,
        return_contacts: bool = False,
        aa_esm: T.Optional[torch.Tensor] = None,  # <--- MODIFIED: Added as optional arg
    ):
        """Runs a forward pass given input tokens. Use `model.infer` to
        run inference from a sequence.

        Args:
            aa (torch.Tensor): Tensor containing indices for the FOLDING trunk (embeddings, structure module).
            mask (torch.Tensor): Binary tensor with 1 meaning position is unmasked and 0 meaning position is masked.
            residx (torch.Tensor): Residue indices of amino acids. Will assume contiguous if not provided.
            masking_pattern (torch.Tensor): Optional masking to pass to the input. Binary tensor of the same size
                as `aa`. Positions with 1 will be masked. ESMFold sometimes produces different samples when
                different masks are provided.
            num_recycles (int): How many recycle iterations to perform. If None, defaults to training max
                recycles, which is 3.
            aa_esm (torch.Tensor, optional): Tensor containing indices for the ESM model. If None, `aa` is used.
        """

        if aa_esm is None:  # <--- MODIFIED: Default to aa if aa_esm not given
            aa_esm = aa

        if mask is None:
            mask = torch.ones_like(aa)

        B = aa.shape[0]
        L = aa.shape[1]
        device = aa.device

        if residx is None:
            residx = torch.arange(L, device=device).expand_as(aa)

        # === ESM ===
        def get_lm_feats(aa_fold_local, aa_esm_local, mask_rate):
            
            esmaa = self._af2_idx_to_esm_idx(aa_esm_local, mask)  # Use aa_esm_local
            random_mask = torch.rand(aa_esm_local.shape, device=device) < mask_rate  # Use aa_esm_local
            if masking_pattern is not None:
                random_mask = random_mask * masking_pattern
            
            esmaa = self._mask_inputs_to_esm(esmaa, random_mask)
            esm_s, lm_output = self._compute_language_model_representations(esmaa, return_contacts=return_contacts)

            # Convert esm_s to the precision used by the trunk and
            # the structure module. These tensors may be a lower precision if, for example,
            # we're running the language model in fp16 precision.
            esm_s = esm_s.to(self.esm_s_combine.dtype)
            esm_s = esm_s.detach()

            # === preprocessing ===
            esm_s = (self.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)
            s_s_0 = self.esm_s_mlp(esm_s)
            s_z_0 = s_s_0.new_zeros(B, L, L, self.cfg.trunk.pairwise_state_dim)
            s_s_0 += self.embedding(aa_fold_local)  # Use aa_fold_local (which is 'aa')

            # seq_feat, pair_feat
            return s_s_0, s_z_0, lm_output

        structure: dict = self.trunk(
            get_lm_feats,
            aa,  # Pass aa (as true_aa)
            residx,
            mask,
            no_recycles=num_recycles,
            mask_rate=mask_rate,
            true_aa_esm=aa_esm,  # Pass aa_esm as optional
        )
        # Documenting what we expect:
        structure = {
            k: v
            for k, v in structure.items()
            if k
            in [
                "s_z",
                "s_s",
                "frames",
                "sidechain_frames",
                "unnormalized_angles",
                "angles",
                "positions",
                "states",
                "lm_output"
            ]
        }

        disto_logits = self.distogram_head(structure["s_z"])
        disto_logits = (disto_logits + disto_logits.transpose(1, 2)) / 2
        structure["distogram_logits"] = disto_logits

        lm_logits = self.lm_head(structure["s_s"])
        structure["lm_logits"] = lm_logits

        structure["aatype"] = aa  # Use aa
        make_atom14_masks(structure)

        for k in [
            "atom14_atom_exists",
            "atom37_atom_exists",
        ]:
            structure[k] *= mask.unsqueeze(-1)
        structure["residue_index"] = residx

        lddt_head = self.lddt_head(structure["states"]).reshape(
            structure["states"].shape[0], B, L, -1, self.lddt_bins
        )
        structure["lddt_head"] = lddt_head
        plddt = categorical_lddt(lddt_head[-1], bins=self.lddt_bins)
        structure["plddt"] = 100 * plddt  # we predict plDDT between 0 and 1, scale to be between 0 and 100.

        ptm_logits = self.ptm_head(structure["s_z"])

        seqlen = mask.type(torch.int64).sum(1)
        structure["ptm_logits"] = ptm_logits
        structure["ptm"] = torch.stack([
            compute_tm(batch_ptm_logits[None, :sl, :sl], max_bins=31, no_bins=self.distogram_bins)
            for batch_ptm_logits, sl in zip(ptm_logits, seqlen)
        ])
        structure.update(
            compute_predicted_aligned_error(
                ptm_logits, max_bin=31, no_bins=self.distogram_bins
            )
        )

        return structure

    @torch.no_grad()
    def infer(
        self,
        sequences: T.Union[str, T.List[str]],
        residx=None,
        masking_pattern: T.Optional[torch.Tensor] = None,
        num_recycles: T.Optional[int] = None,
        residue_index_offset: T.Optional[int] = 512,
        chain_linker: T.Optional[str] = "G" * 25,
        mask_rate: float = 0.0,
        return_contacts: bool = False,
        sequences_esm: T.Optional[T.Union[str, T.List[str]]] = None,  # <--- This argument is already optional
    ):
        """Runs a forward pass given input sequences.

        Args:
            sequences (Union[str, List[str]]): A list of sequences to make predictions for. This is used
                for the folding trunk (embeddings, structure module). Multimers can also be passed in,
                each chain should be separated by a ':' token (e.g. "<chain1>:<chain2>:<chain3>").
            ... (other args) ...
            sequences_esm (Union[str, List[str]], optional): A list of sequences to feed into ESM.
                If None, `sequences` will be used. Must match batch size of `sequences`.
        """
        if isinstance(sequences, str):
            sequences = [sequences]
        
        # Handle sequences_esm
        aatype_esm_tensor = None
        if sequences_esm is not None:
            if isinstance(sequences_esm, str):
                sequences_esm = [sequences_esm]
            
            if len(sequences) != len(sequences_esm):
                raise ValueError(
                    f"Batch size mismatch: {len(sequences)} sequences for folding, "
                    f"but {len(sequences_esm)} sequences for ESM."
                )
            # Encode ESM sequences
            aatype_esm_tensor, mask_esm, _, _, _ = batch_encode_sequences(
                sequences_esm, residue_index_offset, chain_linker
            )
        
        # Encode folding sequences
        aatype, mask, _residx, linker_mask, chain_index = batch_encode_sequences(
            sequences, residue_index_offset, chain_linker
        )
        
        # Check mask consistency if aa_esm is provided
        if aatype_esm_tensor is not None:
             if not torch.all(mask == mask_esm):
                # This is a sanity check. If lengths are different, things will break.
                raise ValueError("Masks for folding and ESM sequences do not match. "
                                    "This implies different sequence lengths or chain breaks.")


        if residx is None:
            residx = _residx
        elif not isinstance(residx, torch.Tensor):
            residx = collate_dense_tensors(residx)

        # Add tensors to device
        aatype, mask, residx, linker_mask = map(
            lambda x: x.to(self.device), (aatype, mask, residx, linker_mask)
        )
        if aatype_esm_tensor is not None:
            aatype_esm_tensor = aatype_esm_tensor.to(self.device)
        
        output = self.forward(
            aatype,  # This is aa
            mask=mask,
            residx=residx,
            masking_pattern=masking_pattern,
            num_recycles=num_recycles,
            mask_rate=mask_rate,
            return_contacts=return_contacts,
            aa_esm=aatype_esm_tensor  # Pass optional tensor (is None if not provided)
        )

        output["atom37_atom_exists"] = output[
            "atom37_atom_exists"
        ] * linker_mask.unsqueeze(2)

        output["mean_plddt"] = (output["plddt"] * output["atom37_atom_exists"]).sum(
            dim=(1, 2)
        ) / output["atom37_atom_exists"].sum(dim=(1, 2))
        output["chain_index"] = chain_index

        return output

    def output_to_pdb(self, output: T.Dict) -> T.List[str]:
        """Returns the pbd (file) string from the model given the model output."""
        return output_to_pdb(output)

    def infer_pdbs(self, seqs: T.List[str], *args, **kwargs) -> T.List[str]:
        """Returns list of pdb (files) strings from the model given a list of input sequences."""
        output = self.infer(seqs, *args, **kwargs)
        return self.output_to_pdb(output)

    def infer_pdb(self, sequence: str, *args, **kwargs) -> str:
        """Returns the pdb (file) string from the model given an input sequence."""
        return self.infer_pdbs([sequence], *args, **kwargs)[0]

    def set_chunk_size(self, chunk_size: T.Optional[int]):
        # This parameter means the axial attention will be computed
        # in a chunked manner. This should make the memory used more or less O(L) instead of O(L^2).
        # It's equivalent to running a for loop over chunks of the dimension we're iterative over,
        # where the chunk_size is the size of the chunks, so 128 would mean to parse 128-lengthed chunks.
        # Setting the value to None will return to default behavior, disable chunking.
        self.trunk.set_chunk_size(chunk_size)

    @property
    def device(self):
        return self.esm_s_combine.device
