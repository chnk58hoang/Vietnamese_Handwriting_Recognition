import torch
import torch.nn as nn


class GreedySearchDecoder(nn.Module):
    def __init__(self, labels, blank=140):
        """
        :param labels: {token:index}
        :param blank: index of blank token
        """
        super(GreedySearchDecoder, self).__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, probs):
        """
        :param probs shape: (batchsize,seq_len,num_classes)
        :return: list of decoded strings
        """
        results = []
        "Get highest tokens probability "
        indices = torch.argmax(probs, dim=-1)

        "Remove duplicate labels"
        indices = torch.unique_consecutive(indices, dim=-1)

        "Remove blank labels"
        for index in indices:
            index = [self.labels[int(i)] for i in index if int(i) != self.blank]
            decode_str = "".join(index)
            results.append(decode_str)

        return results


class BeamSearchDecoder(nn.Module):
    def __init__(self, decoder, blank=140, beam_size=5):
        """
        :param labels: {token:index}
        :param blank: index of blank token
        :param beam_size: max number of hypos to hold after each decode step
        """
        super(BeamSearchDecoder, self).__init__()
        self.decoder = decoder
        self.blank = blank
        self.beam_size = beam_size

    def beamsearch(self, data, k):
        sequences = [[list(), 0.0]]
        # walk over each step in sequence
        for row in data:
            all_candidates = list()
            # expand each current candidate
            for i in range(len(sequences)):
                seq, score = sequences[i]
                for j in range(len(row)):
                    candidate = [seq + [j], score - row[j]]
                    all_candidates.append(candidate)
            # order all candidates by score
            ordered = sorted(all_candidates, key=lambda tup: tup[1])
            # select k best
            sequences = ordered[:k]
        return sequences

    def forward(self, probs):
        """
        :param probs: (batch_size, seq_len, num_classes)
        :return:
        """
        indices = []
        for data in probs:
            k_seqs = self.beamsearch(data, k=self.beam_size)
            indices.append(torch.tensor(k_seqs[-1][0]))
        indices = torch.stack(indices, dim=0)

        "Remove duplicate labels"
        indices = torch.unique_consecutive(indices, dim=-1)

        results = []

        "Remove blank labels"
        for index in indices:
            index = [self.labels[int(i)] for i in index if int(i) != self.blank]
            decode_str = "".join(index)
            results.append(decode_str)

        return results
