# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:35:37 2020

@author: Manuel Camargo
"""
import numpy as np

from support_modules import support as sup


class SuffixPredictor():

    def __init__(self):
        """constructor"""
        self.model = None
        self.spl = dict()
        self.imp = 'Arg Max'
        self.max_trace_size = 0

    def predict(self, params, model, spl, imp):
        self.model = model
        self.spl = spl
        self.max_trace_size = params['max_trace_size']
        self.imp = imp
        predictor = self._get_predictor(params['model_type'])
        sup.print_performed_task('Predicting suffixes')
        return predictor(params)

    def _get_predictor(self, model_type):
        if model_type in ['shared_cat', 'shared_cat_inter']:
            return self._predict_suffix_shared_cat
        elif model_type in ['seq2seq', 'seq2seq_inter']:
            return self._predict_suffix_seq2seq
        else:
            raise ValueError(model_type)

    def _predict_suffix_shared_cat(self, parms):
        """Generate business process suffixes using a keras trained model.
        Args:
            model (keras model): keras trained model.
            prefixes (list): list of prefixes.
            ac_index (dict): index of activities.
            rl_index (dict): index of roles.
            imp (str): method of next event selection.
        """
        # Generation of predictions
        results = list()
        for i, _ in enumerate(self.spl['prefixes']['activities']):
            # Activities and roles input shape(1,5)
            x_ac_ngram = np.append(
                    np.zeros(parms['dim']['time_dim']),
                    np.array(self.spl['prefixes']['activities'][i]),
                    axis=0)[-parms['dim']['time_dim']:].reshape((1, parms['dim']['time_dim']))

            x_rl_ngram = np.append(
                    np.zeros(parms['dim']['time_dim']),
                    np.array(self.spl['prefixes']['roles'][i]),
                    axis=0)[-parms['dim']['time_dim']:].reshape((1, parms['dim']['time_dim']))

            # Times input shape(1,5,1)
            x_t_ngram = np.array([np.append(
                    np.zeros(parms['dim']['time_dim']),
                    np.array(self.spl['prefixes']['times'][i]),
                    axis=0)[-parms['dim']['time_dim']:].reshape((parms['dim']['time_dim'], 1))])
            if parms['model_type'] == 'shared_cat':
                inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram]
            elif parms['model_type'] == 'shared_cat_inter':
                inter_attr_num = self.spl['prefixes']['inter_attr'][i].shape[1]
                x_inter_ngram = np.array([np.append(
                        np.zeros((parms['dim']['time_dim'], inter_attr_num)),
                        self.spl['prefixes']['inter_attr'][i],
                        axis=0)[-parms['dim']['time_dim']:].reshape((parms['dim']['time_dim'], inter_attr_num))])
                inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram, x_inter_ngram]

            pref_size = len(self.spl['prefixes']['activities'][i])
            acum_dur = list()
            ac_suf, rl_suf = list(), list()
            for _  in range(1, self.max_trace_size):
                predictions = self.model.predict(inputs)
                if self.imp == 'Random Choice':
                    # Use this to get a random choice following as PDF the predictions
                    pos = np.random.choice(
                        np.arange(0,len(predictions[0][0])), p=predictions[0][0])
                    pos1 = np.random.choice(
                        np.arange(0, len(predictions[1][0])), p=predictions[1][0])
                elif self.imp == 'Arg Max':
                    # Use this to get the max prediction
                    pos = np.argmax(predictions[0][0])
                    pos1 = np.argmax(predictions[1][0])
                # Activities accuracy evaluation
                x_ac_ngram = np.append(x_ac_ngram, [[pos]], axis=1)
                x_ac_ngram = np.delete(x_ac_ngram, 0, 1)
                x_rl_ngram = np.append(x_rl_ngram, [[pos1]], axis=1)
                x_rl_ngram = np.delete(x_rl_ngram, 0, 1)
                x_t_ngram = np.append(x_t_ngram, [predictions[2]], axis=1)
                x_t_ngram = np.delete(x_t_ngram, 0, 1)
                if parms['model_type'] == 'shared_cat':
                    inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram]
                elif parms['model_type'] == 'shared_cat_inter':
                    x_inter_ngram = np.append(x_inter_ngram, [predictions[3]], axis=1)
                    x_inter_ngram = np.delete(x_inter_ngram, 0, 1)
                    inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram, x_inter_ngram]
                # Stop if the next prediction is the end of the trace
                # otherwise until the defined max_size
                ac_suf.append(pos)
                rl_suf.append(pos1)
                time_pred = predictions[2][0][0]
                if parms['norm_method'] == 'lognorm':
                    acum_dur.append(np.expm1(time_pred * parms['max_dur']))
                else:
                    acum_dur.append(np.rint(time_pred * parms['max_dur']))
    
                time_expected = 0
                if parms['norm_method'] == 'lognorm':
                    time_expected = np.expm1(np.multiply(
                            self.spl['suffixes']['times'][i], parms['max_dur']))
                else:
                    time_expected = np.rint(np.multiply(
                            self.spl['suffixes']['times'][i], parms['max_dur']))
                if parms['index_ac'][pos] == 'end':
                    break
            results.append({
                'ac_pref': self.spl['prefixes']['activities'][i],
                'ac_pred': ac_suf,
                'ac_expect': self.spl['suffixes']['activities'][i],
                'rl_pref': self.spl['prefixes']['roles'][i],
                'rl_pred': rl_suf,
                'rl_expect': self.spl['suffixes']['roles'][i],
                'tm_pref': self.spl['prefixes']['times'][i],
                'tm_pred': acum_dur,
                'tm_expect': time_expected,
                'pref_size': pref_size})
        sup.print_done_task()
        return results

    def _predict_suffix_seq2seq(self, parms):
        """Generate business process suffixes using a keras trained model.
        Args:
            model (keras model): keras trained model.
            prefixes (list): list of prefixes.
            ac_index (dict): index of activities.
            rl_index (dict): index of roles.
            imp (str): method of next event selection.
        """
        # Generation of predictions
        results = list()
        for i in range(0, len(self.spl['prefixes']['activities'])):
            act_prefix = self.spl['prefixes']['activities'][i].reshape(
                    (1, self.spl['prefixes']['activities'][i].shape[0]))
            rl_prefix = self.spl['prefixes']['roles'][i].reshape(
                    (1, self.spl['prefixes']['roles'][i].shape[0]))
            times_prefix = self.spl['prefixes']['times'][i].reshape(
                    (1, self.spl['prefixes']['times'][i].shape[0],
                     self.spl['prefixes']['times'][i].shape[1]))
            if parms['model_type'] == 'seq2seq':
                inputs = [act_prefix, rl_prefix, times_prefix]
            elif parms['model_type'] == 'seq2seq_inter':
                inter_prefix = self.spl['prefixes']['inter_attr'][i].reshape(
                        (1, self.spl['prefixes']['inter_attr'][i].shape[0],
                         self.spl['prefixes']['inter_attr'][i].shape[1]))
                inputs = [act_prefix, rl_prefix, times_prefix, inter_prefix]

            pref_size = len(
                [x for x in self.spl['prefixes']['activities'][i][1:] if x > 0])
            predictions = self.model.predict(inputs)
            if self.imp == 'Random Choice':
                # Use this to get a random choice following as PDF
                act_pred = [np.random.choice(np.arange(0, len(x)), p=x)
                            for x in predictions[0][0]]
                rl_pred = [np.random.choice(np.arange(0, len(x)), p=x)
                           for x in predictions[1][0]]
            elif self.imp == 'Arg Max':
                # Use this to get the max prediction
                act_pred = [np.argmax(x) for x in predictions[0][0]]
                rl_pred = [np.argmax(x) for x in predictions[1][0]]
            # Activities accuracy evaluation
            if act_pred[0] == self.spl['suffixes']['activities'][i][0]:
                results['ac_true'].append(1)
            else:
                results['ac_true'].append(0)
            # Roles accuracy evaluation
            if rl_pred[0] == self.spl['suffixes']['roles'][i][0]:
                results['rl_true'].append(1)
            else:
                results['rl_true'].append(0)
            # Activities suffixes
            idx = self.define_pred_index(act_pred, parms)
            act_pred = act_pred[:idx]
            rl_pred = rl_pred[:idx]
            time_pred = predictions[2][0][:idx]
            if parms['norm_method'] == 'lognorm':
                time_pred = np.expm1(
                        np.multiply(time_pred, parms['max_dur']))
            else:
                time_pred = np.rint(
                        np.multiply(time_pred, parms['max_dur']))

            time_expected = 0
            if parms['norm_method'] == 'lognorm':
                time_expected = np.expm1(np.multiply(
                        self.spl['suffixes']['times'][i], parms['max_dur']))
            else:
                time_expected = np.rint(np.multiply(
                        self.spl['suffixes']['times'][i], parms['max_dur']))
            # Append results
            results.append({
                'ac_pref': self.spl['prefixes']['activities'][i],
                'ac_pred': act_pred,
                'ac_expec': self.spl['suffixes']['activities'][i],
                'rl_pref': self.spl['prefixes']['roles'][i],
                'rl_pred': rl_pred,
                'rl_expec': self.spl['suffixes']['roles'][i],
                'tm_pref': self.spl['prefixes']['times'][i],
                'tm_pred': time_pred,
                'tm_expect': time_expected,
                'pref_size': pref_size})
        sup.print_done_task()
        return results

    @staticmethod
    def define_pred_index(act_pred, parms):
        index = len(act_pred)
        for x in act_pred[::-1]:
            if x == 0:
                index -= 1
            else:
                break
        idx = 0
        for x in act_pred[:index]:
            if parms['index_ac'][x] == 'end':
                idx += 1
                break
            else:
                idx += 1
        return idx
