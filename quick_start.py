# quick_start.py

import h5py
import torch
from transformers import AutoTokenizer

from model.pit.modeling_pit import EvalPITForConditionGeneration
from utils import preprocess_llama_2


def load_protein(h5_file_path, protein_identifier, max_nodes=1022):
    """Load protein sequence and node positions from HDF5."""
    with h5py.File(h5_file_path, 'r') as h5_file:
        group = h5_file[protein_identifier]
        sequence = group['sequence'][()][0].decode()
        node_positions = group['node_position'][()][:max_nodes]
    return sequence, node_positions

def main():
    model_directory    = './ckpts/SEPIT'
    pdb_file_path      = './database/pdb.h5'
    protein_identifier = 'Q59650'

    # Dialogue prompts
    prompts = [
        [
            {"role": "user",      "content": "Explain the function of <bop><protein><eop>."},
            {"role": "assistant", "content": None}
        ],
        [
            {"role": "user",      "content": "What are the short sequence motifs that are found in <bop><protein><eop>?"},
            {"role": "assistant", "content": None}
        ]
    ]

    # Load sequence and node positions
    sequence, node_positions = load_protein(pdb_file_path, protein_identifier)

    # Initialize model on GPU
    device = torch.device('cuda')
    model = EvalPITForConditionGeneration.from_pretrained(model_directory).half().to(device)

    # Initialize tokenizers
    text_tokenizer    = AutoTokenizer.from_pretrained(f'{model_directory}/text_tokenizer')
    text_tokenizer.padding_side = 'left'
    protein_tokenizer = AutoTokenizer.from_pretrained(f'{model_directory}/protein_tokenizer')

    # Encode text prompts
    text_inputs = preprocess_llama_2(prompts, text_tokenizer, max_length=1024)
    text_inputs.pop('labels', None)

    # Encode protein sequence for each prompt
    batch_size = len(prompts)
    protein_inputs = protein_tokenizer(
        [sequence] * batch_size,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=1024
    )

    # Build node_position tensor
    node_positions_tensor = torch.zeros(batch_size, 1024, 3)
    length_positions = node_positions.shape[0]
    node_positions_tensor[:, 1:length_positions+1, :] = torch.from_numpy(node_positions)
    # commented out above line to inference without node positions

    # Combine all inputs
    text_inputs.update({
        'protein_input_ids':      protein_inputs['input_ids'],
        'protein_attention_mask': protein_inputs['attention_mask'],
        'node_position':          node_positions_tensor
    })

    # Move tensors to GPU
    for key, value in text_inputs.items():
        if isinstance(value, torch.Tensor):
            text_inputs[key] = value.to(device)

    # Generate and decode responses
    outputs = model.generate(
        **text_inputs,
        max_new_tokens=512,
        pad_token_id=text_tokenizer.pad_token_id
    )
    responses = text_tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    for response in responses:
        print(response)
        print('\n\n')

if __name__ == '__main__':
    main()