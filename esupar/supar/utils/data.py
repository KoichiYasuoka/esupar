# -*- coding: utf-8 -*-

import torch
import torch.distributed as dist
from esupar.supar.utils.fn import kmeans
from esupar.supar.utils.transform import Batch
from torch.utils.data import DataLoader


class Dataset(torch.utils.data.Dataset):
    r"""
    Dataset that is compatible with :class:`torch.utils.data.Dataset`, serving as a wrapper for manipulating all data fields
    with the operating behaviours defined in :class:`~supar.utils.transform.Transform`.
    The data fields of all the instantiated sentences can be accessed as an attribute of the dataset.

    Args:
        transform (Transform):
            An instance of :class:`~supar.utils.transform.Transform` or its derivations.
            The instance holds a series of loading and processing behaviours with regard to the specific data format.
        data (list[list] or str):
            A list of instances or a filename that will be passed into :meth:`transform.load`.
        kwargs (dict):
            Together with `data`, kwargs will be passed into :meth:`transform.load` to control the loading behaviour.

    Attributes:
        transform (Transform):
            An instance of :class:`~supar.utils.transform.Transform`.
        sentences (list[Sentence]):
            A list of sentences loaded from the data.
            Each sentence includes fields obeying the data format defined in ``transform``.
    """

    def __init__(self, transform, data, **kwargs):
        super(Dataset, self).__init__()

        self.transform = transform
        self.sentences = transform.load(data, **kwargs)

    def __repr__(self):
        s = f"{self.__class__.__name__}("
        s += f"n_sentences={len(self.sentences)}"
        if hasattr(self, 'loader'):
            s += f", n_batches={len(self.loader)}"
        if hasattr(self, 'buckets'):
            s += f", n_buckets={len(self.buckets)}"
        s += ")"

        return s

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return self.sentences[index]

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        return [getattr(sentence, name) for sentence in self.sentences]

    def __setattr__(self, name, value):
        if 'sentences' in self.__dict__ and name in self.sentences[0]:
            # restore the order of sequences in the buckets
            indices = torch.tensor([i for bucket in self.buckets.values() for i in bucket]).argsort()
            for index, sentence in zip(indices, self.sentences):
                setattr(sentence, name, value[index])
        else:
            self.__dict__[name] = value

    def __getstate__(self):
        # only pickle the Transform object and sentences
        return {'transform': self.transform, 'sentences': self.sentences}

    def __setstate__(self, state):
        self.__dict__.update(state)

    def build(self, batch_size, n_buckets=1, shuffle=False, distributed=False):
        # numericalize all fields
        fields = self.transform(self.sentences)
        # NOTE: the final bucket count is roughly equal to n_buckets
        self.buckets = dict(zip(*kmeans([len(s.transformed[fields[0].name]) for s in self], n_buckets)))
        self.loader = DataLoader(dataset=self,
                                 batch_sampler=Sampler(self.buckets, batch_size, shuffle, distributed),
                                 collate_fn=lambda x: Batch(x))
        return self


class Sampler(torch.utils.data.Sampler):
    r"""
    Sampler that supports for bucketization and token-level batchification.

    Args:
        buckets (dict):
            A dict that maps each centroid to indices of clustered sentences.
            The centroid corresponds to the average length of all sentences in the bucket.
        batch_size (int):
            Token-level batch size. The resulting batch contains roughly the same number of tokens as ``batch_size``.
        shuffle (bool):
            If ``True``, the sampler will shuffle both buckets and samples in each bucket. Default: ``False``.
        distributed (bool):
            If ``True``, the sampler will be used in conjunction with :class:`torch.nn.parallel.DistributedDataParallel`
            that restricts data loading to a subset of the dataset.
            Default: ``False``.
    """

    def __init__(self, buckets, batch_size, shuffle=False, distributed=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sizes, self.buckets = zip(*[(size, bucket) for size, bucket in buckets.items()])
        # number of chunks in each bucket, clipped by range [1, len(bucket)]
        self.chunks = [min(len(bucket), max(round(size * len(bucket) / batch_size), 1))
                       for size, bucket in zip(self.sizes, self.buckets)]

        self.rank = dist.get_rank() if distributed else 0
        self.replicas = dist.get_world_size() if distributed else 1
        self.samples = sum(self.chunks) // self.replicas
        self.epoch = 1

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        total, count = 0, 0
        # if `shuffle=True`, shuffle both the buckets and samples in each bucket
        # for distributed training, make sure each process generates the same random sequence at each epoch
        range_fn = torch.arange if not self.shuffle else lambda x: torch.randperm(x, generator=g)
        # TODO: more elegant way to deal with uneven data, which we directly discard right now
        for i in range_fn(len(self.buckets)).tolist():
            split_sizes = [(len(self.buckets[i]) - j - 1) // self.chunks[i] + 1 for j in range(self.chunks[i])]
            # DON'T use `torch.chunk` which may return wrong number of chunks
            for batch in range_fn(len(self.buckets[i])).split(split_sizes):
                if count == self.samples:
                    break
                if total % self.replicas == self.rank:
                    count += 1
                    yield [self.buckets[i][j] for j in batch.tolist()]
                total += 1
        self.epoch += 1

    def __len__(self):
        return self.samples
