import os
import json
import datasets
from datasets import load_dataset, concatenate_datasets, DatasetDict
from tqdm import tqdm
import argparse


def main(args):
    train_teacher_data = load_dataset('json', data_files=os.path.join(args.teacher_file, 'generated_train.jsonl'), split='train')
    train_student_data = load_dataset('json', data_files=os.path.join(args.student_file, 'generated_train.jsonl'), split='train')

    # make sure the pair
    samples = []

    for chosen_raw, rejected_raw in zip(train_teacher_data, train_student_data):
        chosen = [
            {"role": "system", "content": "Put your final answer within \\boxed{}."},
            {"content": chosen_raw['prompt'], "role": "user"},
            {"content": chosen_raw['generated_text'], "role": "assistant"}
        ]
        rejected = [
            {"role": "system", "content": "Put your final answer within \\boxed{}."},
            {"content": rejected_raw['prompt'], "role": "user"},
            {"content": rejected_raw['generated_text'], "role": "assistant"}
        ]
        samples.append({"prompt": rejected_raw['prompt'], "chosen": chosen, "rejected": rejected})

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(f'{args.output_dir}/train.json', 'w') as json_file:
        json.dump(samples, json_file)

    dataset = DatasetDict({
        'train': load_dataset('json', data_files=f'{args.output_dir}/train.json', split='train'),
        'test': load_dataset('json', data_files=f'{args.output_dir}/train.json', split='train').select(range(10)),
    })
    dataset.save_to_disk(args.output_dir)
    print (f"Binarized datasets save to {os.path.join(args.output_dir)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_file", type=str, required=True)
    parser.add_argument("--student_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    main(args)