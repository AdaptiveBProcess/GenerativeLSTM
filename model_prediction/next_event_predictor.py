# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 20:35:53 2020

@author: Manuel Camargo
"""
import numpy as np

from support_modules import support as sup


class NextEventPredictor():

    def __init__(self):
        """constructor"""
        self.model = None
        self.spl = dict()
        self.imp = 'Arg Max'

    def predict(self, params, model, spl, imp):
        self.model = model
        self.spl = spl
        self.imp = imp
        predictor = self._get_predictor(params['model_type'])
        sup.print_performed_task('Predicting next events')
        return predictor(params)

    def _get_predictor(self, model_type):
        if model_type in ['shared_cat', 'shared_cat_rd',
                          'shared_cat_wl', 'shared_cat_inter',
                          'shared_cat_inter_full',
                          'cnn_lstm_inter', 'cnn_lstm_inter_full',
                          'cnn_lstm', 'shared_cat_cx',
                          'shared_cat_city', 'shared_cat_snap']:
            return self._predict_next_event_shared_cat
        else:
            raise ValueError(model_type)

    def _predict_next_event_shared_cat(self, parameters):
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
            x_ac_ngram = (np.append(
                    np.zeros(parameters['dim']['time_dim']),
                    np.array(self.spl['prefixes']['activities'][i]),
                    axis=0)[-parameters['dim']['time_dim']:]
                .reshape((1, parameters['dim']['time_dim'])))

            x_rl_ngram = (np.append(
                    np.zeros(parameters['dim']['time_dim']),
                    np.array(self.spl['prefixes']['roles'][i]),
                    axis=0)[-parameters['dim']['time_dim']:]
                .reshape((1, parameters['dim']['time_dim'])))

            # times input shape(1,5,1)
            x_t_ngram = (np.array([np.append(
                    np.zeros(parameters['dim']['time_dim']),
                    np.array(self.spl['prefixes']['times'][i]),
                    axis=0)[-parameters['dim']['time_dim']:]
                .reshape((parameters['dim']['time_dim'], 1))]))
            # add intercase features if necessary
            if parameters['model_type'] in ['shared_cat', 'cnn_lstm']:
                inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram]
            elif parameters['model_type'] in ['shared_cat_inter',
                                              'shared_cat_inter_full',
                                              'shared_cat_rd',
                                              'shared_cat_wl',
                                              'shared_cat_cx',
                                              'cnn_lstm_inter',
                                              'cnn_lstm_inter_full',
                                              'shared_cat_city',
                                              'shared_cat_snap']:
                # times input shape(1,5,1)
                inter_attr_num = (self.spl['prefixes']['inter_attr'][i]
                                  .shape[1])
                x_inter_ngram = np.array(
                    [np.append(np.zeros((
                        parameters['dim']['time_dim'], inter_attr_num)),
                        self.spl['prefixes']['inter_attr'][i], axis=0)
                        [-parameters['dim']['time_dim']:]
                        .reshape(
                            (parameters['dim']['time_dim'], inter_attr_num))]
                    )
                inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram, x_inter_ngram]
            # predict
            predictions = self.model.predict(inputs)
            if self.imp == 'Random Choice':
                # Use this to get a random choice following as PDF
                pos = np.random.choice(np.arange(0, len(predictions[0][0])),
                                       p=predictions[0][0])
                pos1 = np.random.choice(np.arange(0, len(predictions[1][0])),
                                        p=predictions[1][0])
            elif self.imp == 'Arg Max':
                # Use this to get the max prediction
                pos = np.argmax(predictions[0][0])
                pos1 = np.argmax(predictions[1][0])

            # save results
            results.append({
                'ac_prefix': self.spl['prefixes']['activities'][i],
                'ac_expect': self.spl['next_evt']['activities'][i],
                'ac_pred': pos,
                'rl_prefix': self.spl['prefixes']['roles'][i],
                'rl_expect': self.spl['next_evt']['roles'][i],
                'rl_pred': pos1,
                'tm_prefix': [self.rescale(x, parameters)
                              for x in self.spl['prefixes']['times'][i]],
                'tm_expect': self.rescale(
                    self.spl['next_evt']['times'][i], parameters),
                'tm_pred': self.rescale(predictions[2][0][0], parameters)})
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
