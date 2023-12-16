import json
import os
from typing import List, Any, Tuple

import pandas as pd
import torch
from tap import Tap


def prot_seqs_to_esm_embeddings(
        data: List[Tuple[str, str]],
        model: torch.nn.Module,
        alphabet: Any,
) -> torch.tensor:
    """Convert protein sequences to ESM embeddings."""
    batch_converter = alphabet.get_batch_converter()

    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    avg_seq_representations = []
    for i, tokens_len in enumerate(batch_lens):
        avg_seq_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0).cpu())

    return token_representations.cpu(), torch.stack(avg_seq_representations), batch_lens.tolist()


def main(args):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    df = pd.read_csv(args.input_file_path)

    prot_names = df.prot_name.tolist()
    dna_binding_domains_prot_seqs = [(prot_name, prot_seq) for prot_name, prot_seq in zip(prot_names, df.dna_binding_domain_seq.tolist())]
    prot_seqs = [(prot_name, prot_seq) for prot_name, prot_seq in zip(prot_names, df.prot_seq.tolist())]

    model, alphabet = torch.hub.load("facebookresearch/esm", args.model_name)
    model.eval()

    dbd_aa_embeddings, dbd_avg_embeddings, dbd_seq_lens = prot_seqs_to_esm_embeddings(
        data=dna_binding_domains_prot_seqs, model=model, alphabet=alphabet)

    prot_aa_embeddings, prot_avg_embeddings, prot_seq_lens = prot_seqs_to_esm_embeddings(
        data=prot_seqs, model=model, alphabet=alphabet)

    torch.save(dbd_aa_embeddings, os.path.join(args.output_dir, "dna_binding_domain_aa_prot_embeds.pt"))
    torch.save(dbd_avg_embeddings, os.path.join(args.output_dir, "dna_binding_domain_avg_prot_embeds.pt"))

    torch.save(prot_aa_embeddings, os.path.join(args.output_dir, "prot_aa_prot_embeds.pt"))
    torch.save(prot_avg_embeddings, os.path.join(args.output_dir, "prot_avg_prot_embeds.pt"))

    metadata = {}
    for idx, prot_name in enumerate(prot_names):
        metadata[idx] = {
            "prot_name": prot_name,
            "dna_binding_domain_aa_seq_len": dbd_seq_lens[idx],
            "prot_seq_aa_seq_len": prot_seq_lens[idx],
        }

    with open(os.path.join(args.output_dir, "prot_idx_to_metadata.json"), 'w') as f:
        json.dump(metadata, f)


class ArgParser(Tap):
    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    # file paths for loading data
    input_file_path: str = "tf_prot_seqs.csv"
    output_dir: str = "/tmp/prot-seqs-output/"
    model_name: str = "esm2_t33_650M_UR50D"


if __name__ == "__main__":
    args = ArgParser().parse_args()
    main(args)
