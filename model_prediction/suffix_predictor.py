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

    def predict(self, params, model, spl, imp, vectorizer):
        self.model = model
        self.spl = spl
        self.max_trace_size = params['max_trace_size']
        self.imp = imp
        predictor = self._get_predictor(params['model_type'])
        sup.print_performed_task('Predicting suffixes')
        return predictor(params, vectorizer)

    def _get_predictor(self, model_type):
        # OJO: This is an extension point just incase 
        # a different predictor being neccesary
        return self._predict_suffix_shared_cat

    def _predict_suffix_shared_cat(self, parms, vectorizer):
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
            if vectorizer in ['basic']:
                inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram]
            elif vectorizer in ['inter']:
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
                if vectorizer in ['basic']:
                    inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram]
                elif vectorizer in ['inter']:
                    x_inter_ngram = np.append(x_inter_ngram, [predictions[3]], axis=1)
                    x_inter_ngram = np.delete(x_inter_ngram, 0, 1)
                    inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram, x_inter_ngram]
                # Stop if the next prediction is the end of the trace
                # otherwise until the defined max_size
                ac_suf.append(pos)
                rl_suf.append(pos1)
                acum_dur.append(self.rescale(predictions[2][0][0], parms))
                if parms['index_ac'][pos] == 'end':
                    break
            results.append({
                'ac_pref': self.spl['prefixes']['activities'][i],
                'ac_pred': ac_suf,
                'ac_expect': self.spl['suffixes']['activities'][i],
                'rl_pref': self.spl['prefixes']['roles'][i],
                'rl_pred': rl_suf,
                'rl_expect': self.spl['suffixes']['roles'][i],
                'tm_pref': [self.rescale(x, parms) for x in self.spl['prefixes']['times'][i]],
                'tm_pred': acum_dur,
                'tm_expect': [self.rescale(x, parms) for x in self.spl['suffixes']['times'][i]],
                'pref_size': pref_size})
        sup.print_done_task()
        return results

    @staticmethod
    def rescale(value, parms):
        if parms['norm_method'] == 'lognorm':
            max_value = parms['scale_args']['max_value']
            min_value = parms['scale_args']['min_value']
            value = (value * (max_value - min_value)) + min_value
            value = np.expm1(value)
        elif parms['norm_method'] == 'normal':
            max_value = parms['scale_args']['max_value']
            min_value = parms['scale_args']['min_value']
            value = (value * (max_value - min_value)) + min_value
        elif parms['norm_method'] == 'standard':
            mean = parms['scale_args']['mean']
            std = parms['scale_args']['std']
            value = (value * std) + mean
        elif parms['norm_method'] == 'max':
            max_value = parms['scale_args']['max_value']
            value = np.rint(value * max_value)
        elif parms['norm_method'] is None:
            value = value
        else:
            raise ValueError(parms['norm_method'])
        return value
