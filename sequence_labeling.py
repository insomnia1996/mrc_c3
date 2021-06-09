'''
 @Author: Shuming Ma
 @mail:   shumingma@pku.edu.cn
 @homepage : shumingma.com
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
import modules
import modules.rnn as rnn
from typing import List, Dict, Iterator
from metrics import SequenceAccuracy
from tools import official_tokenization as tokenization

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def sequence_cross_entropy_with_logits(logits: torch.Tensor,
                                       targets: torch.LongTensor,
                                       weights: torch.Tensor,
                                       batch_average: bool = None,
                                       average: str = "batch",
                                       label_smoothing: float = None) -> torch.Tensor:
    """
    Computes the cross entropy loss of a sequence, weighted with respect to
    some user provided weights. Note that the weighting here is not the same as
    in the :func:`torch.nn.CrossEntropyLoss()` criterion, which is weighting
    classes; here we are weighting the loss contribution from particular elements
    in the sequence. This allows loss computations for models which use padding.

    Parameters
    ----------
    logits : ``torch.FloatTensor``, required.
        A ``torch.FloatTensor`` of size (batch_size, sequence_length, num_classes)
        which contains the unnormalized probability for each class.
    targets : ``torch.LongTensor``, required.
        A ``torch.LongTensor`` of size (batch, sequence_length) which contains the
        index of the true class for each corresponding step.
    weights : ``torch.FloatTensor``, required.
        A ``torch.FloatTensor`` of size (batch, sequence_length)
    batch_average : bool, optional, (default = None).
        A bool indicating whether the loss should be averaged across the batch,
        or returned as a vector of losses per batch element.

        .. deprecated:: 0.6.2
           ``batch_average`` was deprecated and replaced with
           the more general ``average`` in version 0.6.2. It will be removed
           in version 0.8.

    average: str, optional (default = "batch")
        If "batch", average the loss across the batches. If "token", average
        the loss across each item in the input. If ``None``, return a vector
        of losses per batch element.
    label_smoothing : ``float``, optional (default = None)
        Whether or not to apply label smoothing to the cross-entropy loss.
        For example, with a label smoothing value of 0.2, a 4 class classifcation
        target would look like ``[0.05, 0.05, 0.85, 0.05]`` if the 3rd class was
        the correct label.

    Returns
    -------
    A torch.FloatTensor representing the cross entropy loss.
    If ``average=="batch"`` or ``average=="token"``, the returned loss is a scalar.
    If ``average is None``, the returned loss is a vector of shape (batch_size,).

    """
    if batch_average is not None:
        # Maintain old behavior
        if batch_average:
            warnings.warn("batch_average=True was deprecated and replaced "
                          "with average='batch' in version 0.6.2. It will be "
                          "removed in version 0.8.", DeprecationWarning)
            average = "batch"
        else:
            warnings.warn("batch_average=False was deprecated and replaced "
                          "with average=None in version 0.6.2. It will be "
                          "removed in version 0.8.", DeprecationWarning)
            average = None
    if average not in {None, "token", "batch"}:
        raise ValueError("Got average f{average}, expected one of "
                         "None, 'token', or 'batch'")

    # shape : (batch * sequence_length, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # shape : (batch * sequence_length, num_classes)
    log_probs_flat = torch.nn.functional.log_softmax(logits_flat, dim=-1)
    # shape : (batch * max_len, 1)
    targets_flat = targets.view(-1, 1).long()

    if label_smoothing is not None and label_smoothing > 0.0:
        num_classes = logits.size(-1)
        smoothing_value = label_smoothing / num_classes
        # Fill all the correct indices with 1 - smoothing value.
        one_hot_targets = torch.zeros_like(log_probs_flat).scatter_(-1, targets_flat, 1.0 - label_smoothing)
        smoothed_targets = one_hot_targets + smoothing_value
        negative_log_likelihood_flat = - log_probs_flat * smoothed_targets
        negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
    else:
        # Contribution to the negative log likelihood only comes from the exact indices
        # of the targets, as the target distributions are one-hot. Here we use torch.gather
        # to extract the indices of the num_classes dimension which contribute to the loss.
        # shape : (batch * sequence_length, 1)
        negative_log_likelihood_flat = - torch.gather(log_probs_flat, dim=1, index=targets_flat)
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())
    # shape : (batch, sequence_length)
    weights = weights.type(logits.type())
    negative_log_likelihood = negative_log_likelihood * weights

    if average == "batch":
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(1) / (weights.sum(1) + 1e-13)
        num_non_empty_sequences = ((weights.sum(1) > 0).type(logits.type()).sum() + 1e-13)
        return per_batch_loss.sum() / num_non_empty_sequences
    elif average == "token":
        return negative_log_likelihood.sum() / (weights.sum() + 1e-13)
    else:
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(1) / (weights.sum(1) + 1e-13)
        return per_batch_loss


tokenizer = tokenization.BertTokenizer(vocab_file=r'./check_points/prev_trained_model/roberta_wwm_ext_large/vocab.txt', do_lower_case=True)
logger.info(len(tokenizer.vocab))#21128

class SequenceLabeling(nn.Module):

    def __init__(self, 
                 emb_size: int,
                 pos_emb_size: int,
                 hidden_size: int,
                 enc_layers: int,
                 dropout: float,
                 bidirectional: bool,
                 vocab_size: int,
                 label_size: int,
                 pos_size: int,
                 rpos_size: int) -> None:

        super(SequenceLabeling, self).__init__()

        self.vocab_size = vocab_size
        self.label_size = label_size
        self.pos_size =pos_size
        self.rpos_size = rpos_size
        self.src_embedding = nn.Embedding(self.vocab_size, emb_size)
        self.lpos_embedding = nn.Embedding(self.pos_size, pos_emb_size)
        self.rpos_embedding = nn.Embedding(self.rpos_size, pos_emb_size)

        self.encoder = rnn.rnn_encoder(emb_size+pos_emb_size*2, hidden_size, enc_layers, dropout, bidirectional)
        self.decoder = nn.Linear(hidden_size, self.label_size)
        self.accuracy = SequenceAccuracy()


    def _get_lengths(self, x: torch.Tensor) -> torch.Tensor:
        lengths = (x > 0).sum(-1)
        return lengths


    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor,
                lpos: torch.Tensor,
                rpos: torch.Tensor) -> Dict[str, torch.Tensor]:


        lengths = self._get_lengths(src)
        lengths, indices = lengths.sort(dim=0, descending=True)
        #logger.info(str(lengths))#实际句子长度*4: [295,291,291,290]
        src = src.index_select(dim=0, index=indices)
        tgt = tgt.index_select(dim=0, index=indices)
        lpos = lpos.index_select(dim=0, index=indices)
        rpos = rpos.index_select(dim=0, index=indices)
        #logger.info("SL input shape: %s, %s." %(src.shape, tgt.shape))#[4,512],[4,512]
        src_embs = torch.cat([self.src_embedding(src),
                              self.lpos_embedding(lpos),
                              self.rpos_embedding(rpos)], dim=-1)
        #lengths = lengths.cpu()
        #src_embs = pack(src_embs, lengths, batch_first=True)
        #logger.info("src_emb"+len(src_embs))
        encode_outputs = self.encoder(src_embs)
        out_logits = self.decoder(encode_outputs['hidden_outputs'])

        seq_mask = (src>0).float()
        #logger.info("SL output shape: %s",str(out_logits.shape))#[batch_size*4,512,2]

        #acc = self.accuracy(predictions=out_logits, gold_labels=tgt, mask=seq_mask)
        loss = sequence_cross_entropy_with_logits(logits=out_logits, 
                                                  targets=tgt,
                                                  weights=seq_mask,
                                                  average='token')
        outputs = {'loss': loss, 'logits': out_logits}
        return outputs
        

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self.accuracy.get_metric(reset)}


    def predict(self, 
                src: Dict[str, torch.Tensor],
                tgt: torch.Tensor,
                key: Dict[str, torch.Tensor],
                lpos: Dict[str, torch.Tensor],
                rpos: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        with torch.no_grad(): 
            src = src['tokens']
            keys, lpos, rpos = key['keys'], lpos['lpos'], rpos['rpos']
            lengths = self._get_lengths(src)
            lengths, indices = lengths.sort(dim=0, descending=True)
            rev_indices = indices.sort()[1]
            src = src.index_select(dim=0, index=indices)
            keys = keys.index_select(dim=0, index=indices)
            lpos = lpos.index_select(dim=0, index=indices)
            rpos = rpos.index_select(dim=0, index=indices)

            src_embs = torch.cat([self.src_embedding(src), 
                                self.key_embedding(keys),
                                self.lpos_embedding(lpos),
                                self.rpos_embedding(rpos)], dim=-1)

            #src_embs = pack(src_embs, lengths, batch_first=True)
            encode_outputs = self.encoder(src_embs)
            out_logits = self.decoder(encode_outputs['hidden_outputs'])
            outputs = out_logits.max(-1)[1]

            outputs = outputs.index_select(dim=0, index=rev_indices)
            src = src.index_select(dim=0, index=rev_indices)

            seq_mask = src>0
            correct_tokens = (tgt.eq(outputs)*seq_mask).sum()
            total_tokens = seq_mask.sum()

            h_tokens = (tgt.eq(outputs)*seq_mask*(tgt.eq(0))).sum()
            r_total_tokens = (tgt.eq(0)*seq_mask).sum()
            p_total_tokens = (outputs.eq(0)*seq_mask).sum()

            output_ids = 1-outputs

            return {'correct': correct_tokens.item(), 'total': total_tokens.item(), 'hit': h_tokens.item(),
                    'r_total': r_total_tokens.item(), 'p_total': p_total_tokens.item(),
                    'output_ids': output_ids.tolist()}