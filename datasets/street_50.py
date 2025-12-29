import os
import random
from collections import defaultdict

from .utils import Datum, DatasetBase, read_json, write_json

template = ['a photo of {}.']


class Street50(DatasetBase):
    """
    Folder-based dataset:

    root/
      street_clean_sample/
        bulky_item_50/
          *.jpg
        clean_50/
          *.jpg
        encampment_50/
          *.jpg
        illegal_dumping_50/
          *.jpg
        overgrown_vegetation_50/
          *.jpg
        split_street_clean_sample.json   (to be generated)
    """

    dataset_dir = "street_clean_sample"
    split_filename = "split_street_clean_sample.json"

    def __init__(self, root, num_shots, seed=1):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.split_path = os.path.join(self.dataset_dir, self.split_filename)
        self.template = template

        # If split doesn't exist, create one automatically
        if not os.path.isfile(self.split_path):
            self._create_split(self.dataset_dir, self.split_path, seed=seed)

        train, val, test = self.read_split(self.split_path, self.dataset_dir)

        # Few-shot handling (same idea as OxfordPets)
        n_shots_val = min(num_shots, 4)
        val = self.generate_fewshot_dataset(val, num_shots=n_shots_val)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)

    @staticmethod
    def _is_image(fn: str) -> bool:
        fn = fn.lower()
        return fn.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))

    @staticmethod
    def _create_split(dataset_dir: str, split_path: str, seed: int = 1,
                      train_ratio: float = 0.7, val_ratio: float = 0.1):
        """
        Build a deterministic split by scanning class folders.
        Remaining fraction becomes test.
        For 50 imgs/class => ~35 train / 5 val / 10 test.
        """
        rng = random.Random(seed)

        classnames = sorted([
            d for d in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, d))
        ])
        if not classnames:
            raise RuntimeError(f"No class folders found under: {dataset_dir}")

        cls2id = {c: i for i, c in enumerate(classnames)}

        split = {"train": [], "val": [], "test": []}

        for c in classnames:
            folder = os.path.join(dataset_dir, c)
            files = [f for f in os.listdir(folder) if StreetCleanSample._is_image(f)]
            files.sort()
            if not files:
                raise RuntimeError(f"No images found in class folder: {folder}")

            rng.shuffle(files)

            n = len(files)
            n_train = int(round(n * train_ratio))
            n_val = int(round(n * val_ratio))
            if n_train + n_val > n:
                n_val = max(0, n - n_train)
            n_test = n - n_train - n_val

            train_files = files[:n_train]
            val_files = files[n_train:n_train + n_val]
            test_files = files[n_train + n_val:]

            label = cls2id[c]

            # Save relative to dataset_dir, e.g. "bulky_item_50/xxx.jpg"
            split["train"].extend([(f"{c}/{f}", label, c) for f in train_files])
            split["val"].extend([(f"{c}/{f}", label, c) for f in val_files])
            split["test"].extend([(f"{c}/{f}", label, c) for f in test_files])

        write_json(split, split_path)
        print(f"Saved split to {split_path}")
        print("Class mapping:")
        for c in classnames:
            print(f"  {cls2id[c]} -> {c}")

    @staticmethod
    def read_split(filepath: str, path_prefix: str):
        """
        Read split JSON:
          {"train":[(relpath,label,classname),...], "val":[...], "test":[...]}
        and convert into Datum objects with absolute impath.
        """
        def _convert(items):
            out = []
            for relpath, label, classname in items:
                impath = os.path.join(path_prefix, relpath)
                out.append(Datum(impath=impath, label=int(label), classname=classname))
            return out

        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        train = _convert(split["train"])
        val = _convert(split["val"])
        test = _convert(split["test"])
        return train, val, test
