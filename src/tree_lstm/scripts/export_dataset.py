from datasets import DatasetDict, Dataset


def main():
    tok_name = "prajjwal1/bert-tiny"
    feat_model = "tree_small"
    dataset = DatasetDict.load_from_disk(f"../../../data/hm_dataset_{tok_name.replace('/', '-')}+{feat_model}")
    for ds in ["train", "test", "validation"]:
        sents = dataset[ds]["sent"]
        labels = dataset[ds]["label"]
        with open(f"../data/ds/ds_{ds}.txt", "w") as f:
            for i in range(len(sents)):
                f.write("\t".join([str(i), sents[i].strip().replace("\n", ""), str(labels[i])]) + "\n")


def xx():
    dataset = Dataset.load_from_disk(f"../../../data/paraphrased_ds/para_set")
    sents = dataset["sent"]
    labels = dataset["label"]
    with open(f"../data_test/ds/ds_test.txt", "w") as f:
        for i in range(len(sents)):
            f.write("\t".join([str(i), sents[i].strip().replace("\n", ""), str(labels[i])]) + "\n")


if __name__ == "__main__":
    # main()
    xx()
