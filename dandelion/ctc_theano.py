# coding:utf-8
__author__ = 'dawei.leng'
__version__ = '1.48'

"""
 Another CTC implemented in theano.
 The `cost()` function of `CTC_Timescale` and `CTC_Logscale` classes return the average NLL over a batch samples given query sequences and score matrices.
 This implementation features:
    1) batch / mask supported.
    2) speed comparable with (~35% slower than) the numba implementation which is the fastest by now.

 Created   :  12, 10, 2015
 Revised   :   8,  3, 2016   ver 1.46
           :   1, 10, 2017   ver 1.47  fix wrong 'static_method' decorators.
           :   3, 20, 2017   ver 1.48  add support for alignment {'pre'/'post'} for `CTC_Logscale` class

 Reference :  [1] Alex Graves, etc., Connectionist temporal classification: labelling unsegmented sequence data with
                  recurrent neural networks, ICML, 2006
              [2] Alex Graves, Supervised sequence labelling with recurrent neural networks, 2014
              [3] Lawrence R. Rabiner, A tutorial on hidden Markov models and selected applications in speech recognition,
                  Proceedings of the IEEE, 1989
              [4] Maas Andrew, etc., https://github.com/amaas/stanford-ctc/blob/master/ctc_fast/ctc-loss/ctc_fast.pyx
              [5] Mohammad Pezeshki, https://github.com/mohammadpz/CTC-Connectionist-Temporal-Classification/blob/master/ctc_cost.py
              [6] Shawn Tan, https://github.com/shawntan/rnn-experiment/blob/master/CTC.ipynb
"""
import theano
from theano import tensor
from theano.ifelse import ifelse
floatX = theano.config.floatX

class CTC_Timescale(object):
    """
    Compute CTC cost using time normalization instead of log scale computation.
    Batch supported.
    To compute the batch cost, use .cost() function below.
    Speed slower than the numba & cython version (~6min vs ~3.9min on word_correction_CTC experiment), much faster than
    the following non-batch version ctc_path_probability().

    B: BATCH_SIZE
    L: query sequence length (maximum length of a batch)
    C: class number
    T: time length (maximum time length of a batch)
    """
    @classmethod
    def cost(self, queryseq, scorematrix, queryseq_mask=None, scorematrix_mask=None, blank_symbol=None, align='pre'):
        """
        Compute CTC cost, using only the forward pass
        :param queryseq: (L, B)
        :param scorematrix: (T, C+1, B)
        :param queryseq_mask: (L, B)
        :param scorematrix_mask: (T, B)
        :param blank_symbol: scalar, = C by default
        :return: negative log likelihood averaged over a batch
        """
        if blank_symbol is None:
            # blank_symbol = scorematrix.shape[1] - 1
            blank_symbol = tensor.cast(scorematrix.shape[1], floatX) - 1.0
        queryseq_padded, queryseq_mask_padded = self._pad_blanks(queryseq, blank_symbol, queryseq_mask)
        results = self.path_probability(queryseq_padded, scorematrix, queryseq_mask_padded, scorematrix_mask, blank_symbol, align)
        NLL = -results[1][-1]                                             # negative log likelihood
        NLL_avg = tensor.mean(NLL)                                        # batch averaged NLL, used as cost
        # if scorematrix_mask is not None:
        #     sm = scorematrix * scorematrix_mask.dimshuffle(0, 'x', 1)
        # else:
        #     sm = scorematrix
        # penalty = tensor.log(sm[:, blank_symbol, :].sum())
        return NLL_avg

    @classmethod
    def path_probability(self, queryseq_padded, scorematrix, queryseq_mask_padded=None, scorematrix_mask=None, blank_symbol=None):
        """
        Compute p(l|x) using only the forward variable
        :param queryseq_padded: (2L+1, B)
        :param scorematrix: (T, C+1, B)
        :param queryseq_mask_padded: (2L+1, B)
        :param scorematrix_mask: (T, B)
        :param blank_symbol: = C by default
        :return:
        """
        if blank_symbol is None:
            # blank_symbol = scorematrix.shape[1] - 1
            blank_symbol = tensor.cast(scorematrix.shape[1], floatX) - 1.0
        if queryseq_mask_padded is None:
                    queryseq_mask_padded = tensor.ones_like(queryseq_padded, dtype=floatX)

        pred_y = self._class_batch_to_labeling_batch(queryseq_padded, scorematrix, scorematrix_mask)  # (T, 2L+1, B), reshaped scorematrix

        r2, r3 = self._recurrence_relation(queryseq_padded, queryseq_mask_padded, blank_symbol)       # r2 (2L+1, 2L+1), r3 (2L+1, 2L+1, B)

        def step(p_curr, p_prev, LLForward, countdown, r2, r3, queryseq_mask_padded):
            """
            [DV, 1-14-2016]: A very weird problem encountered when integrating this CTC implementation into Keras. Before this revision
                             there were no input parameters (r2, r3, queryseq_mask_padded) specified, they just referred to the outer scope ones.
                             However, this will cause the CTC integrated within Keras producing inaccurate loss value, meanwhile when compiled
                             as a separate function, the returned ctc loss value is accurate anyway. But if with these 3 parameters added as
                             input, the problem vanished. This took me two days to find this remedy. I suspect this'd be the bug of theano.
            :param p_curr:     (2L+1, B), one column of scorematrix
            :param p_prev:     (B, 2L+1)
            :param LLForward:  (B, 1)
            :param countdown:  scalar
            :param r2:         (2L+1, 2L+1)
            :param r3:         (2L+1, 2L+1, B)
            :param queryseq_mask_padded: (2L+1, B)
            :return:
            """
            dotproduct = (p_prev + tensor.dot(p_prev, r2) +                                           # tensor.dot(p_prev, r2) = alpha(t-1, u-1)
                          (p_prev.dimshuffle(1, 'x', 0) * r3).sum(axis=0).T)                          # = alpha(t-1, u-2) conditionally
            p_curr = p_curr.T * dotproduct
            if queryseq_mask_padded is not None:
                p_curr *= queryseq_mask_padded.T                                                      # (B, 2L+1) * (B, 2L+1) * (B, 2L+1) = (B, 2L+1)
            start = tensor.max([0, queryseq_padded.shape[0] - 2 * countdown])
            mask = tensor.concatenate([tensor.zeros([queryseq_padded.shape[1], start]),
                                       tensor.ones([queryseq_padded.shape[1], queryseq_padded.shape[0] - start])], axis=1)
            p_curr *= mask
            c_batch = p_curr.sum(axis=1, keepdims=True)                                               # (B, 1)
            p_curr /= c_batch
            LLForward += tensor.log(c_batch)
            countdown -= 1
            return p_curr, LLForward, countdown                                                       # (B, 2L+1), (B, 1), scalar

        results, _ = theano.scan(
                step,
                sequences=[pred_y],                                                                   # scan only work on the first dimension
                outputs_info=[tensor.eye(queryseq_padded.shape[0])[0] * tensor.ones(queryseq_padded.T.shape),
                              tensor.unbroadcast(tensor.zeros([queryseq_padded.shape[1], 1]), 1), scorematrix.shape[0]],
                non_sequences=[r2, r3, queryseq_mask_padded])
        return results

    @classmethod
    def best_path_decode(self, scorematrix, scorematrix_mask=None, blank_symbol=None):
        """
        Computes the best path by simply choosing most likely label at each timestep
        :param scorematrix: (T, C+1, B)
        :param scorematrix_mask: (T, B)
        :param blank_symbol: = C by default
        :return: resultseq (T, B), resultseq_mask(T, B)
        Speed much slower than pure python version (normally ~40 times on HTR tasks)
        """
        bestlabels = tensor.argmax(scorematrix, axis=1)    # (T, B)
        T, Cp, B = scorematrix.shape
        resultseq, resultseq_mask = tensor.zeros([T, B], dtype=scorematrix.dtype)-1, tensor.zeros([T, B], dtype=scorematrix.dtype)
        if blank_symbol is None:
            # blank_symbol = Cp - 1.0
            blank_symbol = tensor.cast(Cp, floatX) - 1.0
        if scorematrix_mask is None:
            scorematrix_mask = tensor.ones([T, B], dtype=scorematrix.dtype)

        def step(labelseq, labelseq_mask, idx, resultseq, resultseq_mask, blank_symbol):
            seqlen = tensor.cast(labelseq_mask.sum(), 'int32')
            labelseq = self._remove_adjdup(labelseq[0:seqlen])
            labelseq = self._remove_value(labelseq, blank_symbol)
            seqlen2 = labelseq.size
            resultseq = tensor.set_subtensor(resultseq[0:seqlen2, idx], labelseq)
            resultseq_mask = tensor.set_subtensor(resultseq_mask[0:seqlen2, idx], tensor.ones_like(labelseq))
            idx += 1
            return idx, resultseq, resultseq_mask

        outputs, updates = theano.scan(fn = step,
                                       sequences=[bestlabels.T, scorematrix_mask.T],
                                       outputs_info=[0, resultseq, resultseq_mask],
                                       non_sequences=[blank_symbol],
                                       name='decode_scan')
        resultseq, resultseq_mask = outputs[1][-1], outputs[2][-1]
        return resultseq, resultseq_mask

    @classmethod
    def calc_CER(self, resultseq, targetseq, resultseq_mask=None, targetseq_mask=None):
        """
        Calculate the character error rate (CER) given ground truth 'targetseq' and CTC decoding output 'resultseq'
        :param resultseq (T1,  B)
        :param resultseq_mask (T1, B)
        :param targetseq (T2,  B)
        :param targetseq_mask (T2, B)
        :return: CER scalar
        """
        if resultseq_mask is None:
            resultseq_mask = tensor.ones_like(resultseq)
        if targetseq_mask is None:
            targetseq_mask = tensor.ones_like(targetseq)

        def step(result_seq, target_seq, result_seq_mask, target_seq_mask, TE, TG):
            L1 = tensor.cast(result_seq_mask.sum(), 'int32')
            L2 = tensor.cast(target_seq_mask.sum(), 'int32')
            d = self._editdist(result_seq[0:L1], target_seq[0:L2])
            TE += d
            TG += target_seq_mask.sum()
            return TE, TG

        outputs, updates = theano.scan(fn=step,
                                       sequences=[resultseq.T, targetseq.T, resultseq_mask.T, targetseq_mask.T],
                                       outputs_info=[tensor.zeros(1), tensor.zeros(1)],
                                       name='calc_CER')
        TE, TG = outputs[0][-1], outputs[1][-1]
        CER = TE/TG
        return CER, TE, TG

    @staticmethod
    def _remove_value(x, value):
        """
        Remove certain valued elements from a vector
        x: vector (must); value: scalar
        return a vector with all elements = 'value' removed
        """
        return (x - value).nonzero_values() + value

    @staticmethod
    def _remove_adjdup(x):
        """
        Remove adjacent duplicate items of a vector
        x: vector
        return a vector with adjacent duplicate items removed, for example [1,2,2,2,3,3,4] -> [1,2,3,4]
        """
        def update(x, nondup, idx):
            nondup = tensor.switch(tensor.eq(nondup[idx], x), nondup, tensor.set_subtensor(nondup[idx + 1], x))  # tensor.switch is much faster than ifelse
            idx = tensor.switch(tensor.eq(nondup[idx], x), idx, idx + 1)
            return nondup, idx
        nondup = x
        idx = tensor.as_tensor_variable(0)
        idx = tensor.cast(idx, 'int32')
        result, updates = theano.scan(fn = update, sequences=x, outputs_info=[nondup, idx], name='remove_adjdup')
        nondup = result[0][-1]
        idx = result[1][-1]
        return nondup[0:idx+1]

    @staticmethod
    def _editdist(s, t):
        """
        Levenshtein's edit distance function
        :param s: vector, source string
        :param t: vector, target string
        :return:  edit distance, scalar
        """
        def update(x, previous_row):
            current_row = previous_row + 1
            current_row = tensor.set_subtensor(current_row[1:], tensor.minimum(current_row[1:], tensor.add(previous_row[:-1], tensor.neq(target,x))))
            current_row = tensor.set_subtensor(current_row[1:], tensor.minimum(current_row[1:], current_row[0:-1] + 1))
            return current_row
        source, target = ifelse(tensor.lt(s.shape[0], t.shape[0]), (t, s), (s, t))
        previous_row = tensor.arange(target.size + 1, dtype=theano.config.floatX)
        result, updates = theano.scan(fn=update, sequences=source, outputs_info=previous_row, name='editdist')
        return result[-1,-1]

    @staticmethod
    def _pad_blanks(queryseq, blank_symbol, queryseq_mask=None):
        """
        Pad queryseq and corresponding queryseq_mask with blank symbol
        :param queryseq  (L, B)
        :param queryseq_mask (L, B)
        :param blank_symbol  scalar, must be float type!
        :return queryseq_padded, queryseq_mask_padded, both with shape (2L+1, B)
        """
        # for queryseq
        queryseq_extended = queryseq.dimshuffle(1, 0, 'x')                              # (L, B) -> (B, L, 1)
        blanks = tensor.zeros_like(queryseq_extended) + blank_symbol                    # (B, L, 1)
        concat = tensor.concatenate([queryseq_extended, blanks], axis=2)                # concat.shape = (B, L, 2)
        res = concat.reshape((concat.shape[0], concat.shape[1] * concat.shape[2])).T    # res.shape = (2L, B), the reshape will cause the last 2 dimensions interlace
        begining_blanks = tensor.zeros((1, res.shape[1])) + blank_symbol                # (1, B)
        queryseq_padded = tensor.concatenate([begining_blanks, res], axis=0)            # (1+2L, B)
        # for queryseq_mask
        if queryseq_mask is not None:
            queryseq_mask_extended = queryseq_mask.dimshuffle(1, 0, 'x')                          # (L, B) -> (B, L, 1)
            concat = tensor.concatenate([queryseq_mask_extended, queryseq_mask_extended], axis=2) # concat.shape = (B, L, 2)
            res = concat.reshape((concat.shape[0], concat.shape[1] * concat.shape[2])).T
            begining_blanks = tensor.ones((1, res.shape[1]), dtype=floatX)
            queryseq_mask_padded = tensor.concatenate([begining_blanks, res], axis=0)
        else:
            queryseq_mask_padded = None
        return queryseq_padded, queryseq_mask_padded

    @staticmethod
    def _class_batch_to_labeling_batch(queryseq_padded, scorematrix, scorematrix_mask=None):
        """
        Convert dimension 'class' of scorematrix to 'label'
        :param queryseq_padded: (2L+1, B)
        :param scorematrix: (T, C+1, B)
        :param scorematrix_mask: (T, B)
        :return: (T, 2L+1, B)
        """
        if scorematrix_mask is not None:
            scorematrix = scorematrix * scorematrix_mask.dimshuffle(0, 'x', 1)                   # (T, C+1, B) * (T, 1, B)
        batch_size = scorematrix.shape[2]  # = B
        res = scorematrix[:, queryseq_padded.astype('int32'), tensor.arange(batch_size)]         # (T, 2L+1, B), indexing each row of scorematrix with queryseq_padded
        return res

    @staticmethod
    def _recurrence_relation(queryseq_padded, queryseq_mask_padded=None, blank_symbol=None):
        """
        Generate structured matrix r2 & r3 for dynamic programming recurrence
        :param queryseq_padded: (2L+1, B)
        :param queryseq_mask_padded: (2L+1, B)
        :param blank_symbol: = C
        :return: r2 (2L+1, 2L+1), r3 (2L+1, 2L+1, B)
        """
        L2 = queryseq_padded.shape[0]                                                           # = 2L+1
        blanks = tensor.zeros((2, queryseq_padded.shape[1])) + blank_symbol                     # (2, B)
        ybb = tensor.concatenate((queryseq_padded, blanks), axis=0).T                           # (2L+3, B) -> (B, 2L+3)
        sec_diag = tensor.neq(ybb[:, :-2], ybb[:, 2:]) * tensor.eq(ybb[:, 1:-1], blank_symbol)  # (B, 2L+1)
        if queryseq_mask_padded is not None:
            sec_diag *= queryseq_mask_padded.T
        r2 = tensor.eye(L2, k=1)                                                                # upper diagonal matrix (2L+1, 2L+1)
        r3 = tensor.eye(L2, k=2).dimshuffle(0, 1, 'x') * sec_diag.dimshuffle(1, 'x', 0)         # (2L+1, 2L+1, B)
        return r2, r3

class CTC_Logscale(CTC_Timescale):
    """
    This implementation uses log scale computation.
    Batch supported. Note the log scale computation used to produce imprecise CTC cost in Theano (path probability).
    [Credits to Mohammad Pezeshki, https://github.com/mohammadpz/CTC-Connectionist-Temporal-Classification]
    B: BATCH_SIZE
    L: query sequence length (maximum length of a batch)
    C: class number
    T: time length (maximum time length of a batch)
    """
    @classmethod
    def cost(self, queryseq, scorematrix, queryseq_mask=None, scorematrix_mask=None, blank_symbol=None, align='pre'):
        """
        Compute CTC cost, using only the forward pass
        :param queryseq: (L, B)
        :param scorematrix: (T, C+1, B)
        :param queryseq_mask: (L, B)
        :param scorematrix_mask: (T, B)
        :param blank_symbol: scalar, = C by default
        :param align: string, {'pre'/'post'}, indicating how input samples are aligned in one batch
        :return: negative log likelihood averaged over a batch
        """
        if blank_symbol is None:
            # blank_symbol = scorematrix.shape[1] - 1.0
            blank_symbol = tensor.cast(scorematrix.shape[1], floatX) - 1.0
        queryseq_padded, queryseq_mask_padded = self._pad_blanks(queryseq, blank_symbol, queryseq_mask)
        NLL, alphas = self.path_probability(queryseq_padded, scorematrix, queryseq_mask_padded, scorematrix_mask, blank_symbol, align)
        NLL_avg = tensor.mean(NLL)
        return NLL_avg

    @classmethod
    def path_probability(self, queryseq_padded, scorematrix, queryseq_mask_padded=None, scorematrix_mask=None, blank_symbol=None, align='pre'):
        """
        Compute p(l|x) using only the forward variable and log scale
        :param queryseq_padded: (2L+1, B)
        :param scorematrix: (T, C+1, B)
        :param queryseq_mask_padded: (2L+1, B)
        :param scorematrix_mask: (T, B)
        :param blank_symbol: = C by default
        :return:
        """
        if blank_symbol is None:
            # blank_symbol = scorematrix.shape[1] - 1.0
            blank_symbol = tensor.cast(scorematrix.shape[1], floatX) - 1.0
        if queryseq_mask_padded is None:
            queryseq_mask_padded = tensor.ones_like(queryseq_padded, dtype=floatX)
        if scorematrix_mask is None:
            scorematrix_mask = tensor.ones([scorematrix.shape[0], scorematrix.shape[2]])

        pred_y = self._class_batch_to_labeling_batch(queryseq_padded, scorematrix, scorematrix_mask)  # (T, 2L+1, B), reshaped scorematrix
        r2, r3 = self._recurrence_relation(queryseq_padded, queryseq_mask_padded, blank_symbol)       # r2 (2L+1, 2L+1), r3 (2L+1, 2L+1, B)

        def step(p_curr, p_prev):
            p1 = p_prev
            p2 = self._log_dot_matrix(p1, r2)
            p3 = self._log_dot_tensor(p1, r3)
            p123 = self._log_add(p3, self._log_add(p1, p2))
            return p_curr.T + p123 + self._epslog(queryseq_mask_padded.T)

        alphas, _ = theano.scan(
                step,
                sequences=[self._epslog(pred_y)],
                outputs_info=[self._epslog(tensor.eye(queryseq_padded.shape[0])[0] * tensor.ones(queryseq_padded.T.shape))])

        B = alphas.shape[1]
        LL = tensor.sum(queryseq_mask_padded, axis=0, dtype='int32')
        if align == 'pre':
            TL = tensor.sum(scorematrix_mask, axis=0, dtype='int32')
            NLL = -self._log_add(alphas[TL - 1, tensor.arange(B), LL - 1],
                                 alphas[TL - 1, tensor.arange(B), LL - 2])
        else:  # align == 'post'
            NLL = -self._log_add(alphas[-1, tensor.arange(B), LL - 1],
                                 alphas[-1, tensor.arange(B), LL - 2])
        return NLL, alphas

    @staticmethod
    def _epslog(x):
        return tensor.cast(tensor.log(tensor.clip(x, 1E-12, 1E12)),
                           theano.config.floatX)

    @staticmethod
    def _log_add(a, b):
        max_ = tensor.maximum(a, b)
        return max_ + tensor.log1p(tensor.exp(a + b - 2 * max_))

    @staticmethod
    def _log_dot_matrix(x, z):
        inf = 1E12
        log_dot = tensor.dot(x, z)
        zeros_to_minus_inf = (z.max(axis=0) - 1) * inf
        return log_dot + zeros_to_minus_inf

    @staticmethod
    def _log_dot_tensor(x, z):
        inf = 1E12
        log_dot = (x.dimshuffle(1, 'x', 0) * z).sum(axis=0).T
        zeros_to_minus_inf = (z.max(axis=0) - 1) * inf
        return log_dot + zeros_to_minus_inf.T


def ctc_path_probability(scorematrix, queryseq, blank):
    """
    Compute path probability based on CTC algorithm, only forward pass is used.
    Batch not supported, for batch version, refer to the CTC class above
    Speed much slower than the numba & cython version (51.5min vs ~3.9min on word_correction_CTC experiment)
    :param scorematrix: (T, C+1)
    :param queryseq:    (L, 1)
    :param blank:       scalar, blank symbol
    :return: (NLL, alphas), NLL > 0 (smaller is better, = -log(p(l|x)); alphas is the forward variable)
    """

    def update_s(s, alphas, scorematrix, queryseq, blank, t):
        l = (s - 1) // 2
        alphas = ifelse(tensor.eq(s % 2, 0),
                        ifelse(tensor.eq(s, 0),
                               tensor.set_subtensor(alphas[s, t], alphas[s, t - 1] * scorematrix[blank, t]),
                               tensor.set_subtensor(alphas[s, t],
                                                    (alphas[s, t - 1] + alphas[s - 1, t - 1]) * scorematrix[blank, t]),
                               name='for_blank_symbol'),
                        ifelse(tensor.or_(tensor.eq(s, 1), tensor.eq(queryseq[l], queryseq[l - 1])),
                               tensor.set_subtensor(alphas[s, t],
                                                    (alphas[s, t - 1] + alphas[s - 1, t - 1]) * scorematrix[
                                                        queryseq[l], t]),
                               tensor.set_subtensor(alphas[s, t],
                                                    (alphas[s, t - 1] + alphas[s - 1, t - 1] + alphas[s - 2, t - 1]) *
                                                    scorematrix[queryseq[l], t]),
                               name='for_same_label_twice'))
        return alphas

    def update_t(t, LLForward, alphas, scorematrix, queryseq, blank, T, L2):
        start = tensor.max([0, L2 - 2 * (T - t)])
        end = tensor.min([2 * t + 2, L2])
        s = tensor.arange(start, end)
        results, _ = theano.scan(fn=update_s, sequences=[s], non_sequences=[scorematrix, queryseq, blank, t],
                                 outputs_info=[alphas], name='scan_along_s')
        alphas = results[-1]
        c = tensor.sum(alphas[start:end, t])
        c = tensor.max([1e-15, c])
        alphas = tensor.set_subtensor(alphas[start:end, t], alphas[start:end, t] / c)
        LLForward += tensor.log(c)
        return LLForward, alphas

    L = queryseq.shape[0]                                                 # Length of label sequence
    L2 = 2 * L + 1                                                        # Length of label sequence padded with blanks
    T = scorematrix.shape[1]                                              # time length
    alphas = tensor.zeros((L2, T))
    # Initialize alphas and forward pass
    alphas = tensor.set_subtensor(alphas[[0, 1], 0], scorematrix[[blank, queryseq[0]], 0])
    c = tensor.sum(alphas[:, 0])
    alphas = tensor.set_subtensor(alphas[:, 0], alphas[:, 0] / c)
    LLForward = tensor.log(c)
    t = tensor.arange(1, T)
    results, _ = theano.scan(fn=update_t, sequences=[t], non_sequences=[scorematrix, queryseq, blank, T, L2],
                             outputs_info=[LLForward, alphas], name='scan_along_t')
    NLL, alphas = ifelse(tensor.gt(T, 1), (-results[0][-1], results[1][-1]), (-LLForward, alphas))
    return NLL, alphas



